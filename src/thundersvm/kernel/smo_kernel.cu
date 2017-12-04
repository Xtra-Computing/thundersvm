//
// Created by jiashuai on 17-9-21.
//
#include "thundersvm/kernel/smo_kernel.h"

#ifdef USE_CUDA
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/par.h>
namespace svm_kernel {

    __device__ int get_block_min(const float *values, int *index) {
        int tid = threadIdx.x;
        index[tid] = tid;
        __syncthreads();
        //block size is always the power of 2
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                if (values[index[tid + offset]] <= values[index[tid]]) {
                    index[tid] = index[tid + offset];
                }
            }
            __syncthreads();
        }
        return index[0];
    }


    __global__ void
    c_smo_solve_kernel(const int *label, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                       const int *working_set,
                       int ws_size,
                       float Cp, float Cn, const float *k_mat_rows, const float *k_mat_diag, int row_len,
                       float_type eps,
                       float_type *diff, int max_iter) {
        //"row_len" equals to the number of instances in the original training dataset.
        //allocate shared memory
        extern __shared__ int shared_mem[];
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        int tid = threadIdx.x;
        int wsi = working_set[tid];
        kd[tid] = k_mat_diag[wsi];
        float y = label[wsi];
        float f = f_val[wsi];
        float a = alpha[wsi];
        float aold = a;
        __syncthreads();
        float local_eps;
        int numOfIter = 0;
        while (1) {
            //select fUp and fLow
            if (is_I_up(a, y, Cp, Cn))
                f_val2reduce[tid] = f;
            else
                f_val2reduce[tid] = INFINITY;
            int i = get_block_min(f_val2reduce, f_idx2reduce);
            float up_value = f_val2reduce[i];
            float kIwsI = k_mat_rows[row_len * i + wsi];//K[i, wsi]
            __syncthreads();

            if (is_I_low(a, y, Cp, Cn))
                f_val2reduce[tid] = -f;
            else
                f_val2reduce[tid] = INFINITY;
            int j1 = get_block_min(f_val2reduce, f_idx2reduce);
            float low_value = -f_val2reduce[j1];

            float local_diff = low_value - up_value;
            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                if (tid == 0) {
                    diff[0] = local_diff;
                }
            }

            if (local_diff < local_eps) {
                alpha[wsi] = a;
                alpha_diff[tid] = -(a - aold) * y;
                break;
            }
            __syncthreads();

            //select j2 using second order heuristic
            if (-up_value > -f && (is_I_low(a, y, Cp, Cn))) {
                float aIJ = kd[i] + kd[tid] - 2 * kIwsI;
                float bIJ = -up_value + f;
                f_val2reduce[tid] = -bIJ * bIJ / aIJ;
            } else
                f_val2reduce[tid] = INFINITY;
            int j2 = get_block_min(f_val2reduce, f_idx2reduce);

            //update alpha
            if (tid == i)
                *alpha_i_diff = y > 0 ? Cp - a : a;
            if (tid == j2)
                *alpha_j_diff = min(y > 0 ? a : Cn - a, (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
            __syncthreads();
            float l = min(*alpha_i_diff, *alpha_j_diff);

            if (tid == i)
                a += l * y;
            if (tid == j2)
                a -= l * y;

            //update f
            float kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
            f -= l * (kJ2wsI - kIwsI);
            numOfIter++;
            if (numOfIter > max_iter) break;
        }
    }


    __global__ void
    nu_smo_solve_kernel(const int *label, float_type *f_values, float_type *alpha, float_type *alpha_diff,
                        const int *working_set,
                        int ws_size, float C, const float *k_mat_rows, const float *k_mat_diag, int row_len,
                        float_type eps,
                        float_type *diff, int max_iter) {
        //"row_len" equals to the number of instances in the original training dataset.
        //allocate shared memory
        extern __shared__ int shared_mem[];
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        int tid = threadIdx.x;
        int wsi = working_set[tid];
        kd[tid] = k_mat_diag[wsi];
        float y = label[wsi];
        float f = f_values[wsi];
        float a = alpha[wsi];
        float aold = a;
        __syncthreads();
        float local_eps;
        int numOfIter = 0;
        while (1) {
            //select I_up (y=+1)
            if (y > 0 && a < C)
                f_val2reduce[tid] = f;
            else
                f_val2reduce[tid] = INFINITY;
            __syncthreads();
            int ip = get_block_min(f_val2reduce, f_idx2reduce);
            float up_value_p = f_val2reduce[ip];
            float kIpwsI = k_mat_rows[row_len * ip + wsi];//K[i, wsi]
            __syncthreads();

            //select I_up (y=-1)
            if (y < 0 && a > 0)
                f_val2reduce[tid] = f;
            else
                f_val2reduce[tid] = INFINITY;
            int in = get_block_min(f_val2reduce, f_idx2reduce);
            float up_value_n = f_val2reduce[in];
            float kInwsI = k_mat_rows[row_len * in + wsi];//K[i, wsi]
            __syncthreads();

            //select I_low (y=+1)
            if (y > 0 && a > 0)
                f_val2reduce[tid] = -f;
            else
                f_val2reduce[tid] = INFINITY;
            int j1p = get_block_min(f_val2reduce, f_idx2reduce);
            float low_value_p = -f_val2reduce[j1p];
            __syncthreads();

            //select I_low (y=-1)
            if (y < 0 && a < C)
                f_val2reduce[tid] = -f;
            else
                f_val2reduce[tid] = INFINITY;
            int j1n = get_block_min(f_val2reduce, f_idx2reduce);
            float low_value_n = -f_val2reduce[j1n];

            float local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);

            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                if (tid == 0) {
                    diff[0] = local_diff;
                }
            }

            if (local_diff < local_eps) {
                alpha[wsi] = a;
                alpha_diff[tid] = -(a - aold) * y;
                break;
            }
            __syncthreads();

            //select j2p using second order heuristic
            if (-up_value_p > -f && y > 0 && a > 0) {
                float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI;
                float bIJ = -up_value_p + f;
                f_val2reduce[tid] = -bIJ * bIJ / aIJ;
            } else
                f_val2reduce[tid] = INFINITY;
            int j2p = get_block_min(f_val2reduce, f_idx2reduce);
            float f_val_j2p = f_val2reduce[j2p];
            __syncthreads();

            //select j2n using second order heuristic
            if (-up_value_n > -f && y < 0 && a < C) {
                float aIJ = kd[in] + kd[tid] - 2 * kInwsI;
                float bIJ = -up_value_n + f;
                f_val2reduce[tid] = -bIJ * bIJ / aIJ;
            } else
                f_val2reduce[tid] = INFINITY;
            int j2n = get_block_min(f_val2reduce, f_idx2reduce);

            int i, j2;
            float up_value;
            float kIwsI;
            if (f_val_j2p < f_val2reduce[j2n]) {
                i = ip;
                j2 = j2p;
                up_value = up_value_p;
                kIwsI = kIpwsI;
            } else {
                i = in;
                j2 = j2n;
                kIwsI = kInwsI;
                up_value = up_value_n;
            }
            //update alpha
            if (tid == i)
                *alpha_i_diff = y > 0 ? C - a : a;
            if (tid == j2)
                *alpha_j_diff = min(y > 0 ? a : C - a, (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
            __syncthreads();
            float l = min(*alpha_i_diff, *alpha_j_diff);

            if (tid == i)
                a += l * y;
            if (tid == j2)
                a -= l * y;

            //update f
            float kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
            f -= l * (kJ2wsI - kIwsI);
            numOfIter++;
            if (numOfIter > max_iter) break;
        }
    }

    void
    c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                SyncArray<float_type> &alpha_diff,
                const SyncArray<int> &working_set, float_type Cp, float_type Cn, const SyncArray<float_type> &k_mat_rows,
                const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                int max_iter) {
        size_t ws_size = working_set.size();
        size_t smem_size = ws_size * sizeof(float_type) * 3 + 2 * sizeof(float);
        c_smo_solve_kernel << < 1, ws_size, smem_size >> >
                                            (y.device_data(), f_val.device_data(), alpha.device_data(), alpha_diff.device_data(),
                                                    working_set.device_data(), ws_size, Cp, Cn, k_mat_rows.device_data(), k_mat_diag.device_data(),
                                                    row_len, eps, diff.device_data(), max_iter);
    }

    void nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                      SyncArray<float_type> &alpha_diff,
                      const SyncArray<int> &working_set, float_type C, const SyncArray<float_type> &k_mat_rows,
                      const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                      int max_iter) {
        size_t ws_size = working_set.size();
        size_t smem_size = ws_size * sizeof(float_type) * 3 + 2 * sizeof(float);
        nu_smo_solve_kernel << < 1, ws_size, smem_size >> >
                                             (y.device_data(), f_val.device_data(), alpha.device_data(), alpha_diff.device_data(),
                                                     working_set.device_data(), ws_size, C, k_mat_rows.device_data(), k_mat_diag.device_data(),
                                                     row_len, eps, diff.device_data(), max_iter);
    }

    __global__ void
    update_f_kernel(float_type *f, int ws_size, const float_type *alpha_diff, const float_type *k_mat_rows,
                    int n_instances) {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        KERNEL_LOOP(idx, n_instances) {//one thread to update multiple fvalues.
            float_type sum_diff = 0;
            for (int i = 0; i < ws_size; ++i) {
                float_type d = alpha_diff[i];
                if (d != 0) {
                    sum_diff += d * k_mat_rows[i * n_instances + idx];
                }
            }
            f[idx] -= sum_diff;
        }
    }

    void
    update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff, const SyncArray<float_type> &k_mat_rows,
             int n_instances) {
        SAFE_KERNEL_LAUNCH(update_f_kernel, f.device_data(), alpha_diff.size(), alpha_diff.device_data(),
                           k_mat_rows.device_data(), n_instances);
    }

    void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort) {
        thrust::sort_by_key(thrust::cuda::par, f_val2sort.device_data(), f_val2sort.device_data() + f_val2sort.size(),
                            f_idx2sort.device_data(), thrust::less<float_type>());
    }
}
#endif
