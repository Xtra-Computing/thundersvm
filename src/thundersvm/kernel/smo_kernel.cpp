
//
// Created by jiashuai on 17-11-7.
//

#include <thundersvm/kernel/smo_kernel.h>

namespace svm_kernel {
    int get_min_idx(const float *values, int size) {
        int min_idx = 0;
        float min = INFINITY;
        for (int i = 0; i < size; ++i) {
            if (values[i] < min) {
                min = values[i];
                min_idx = i;
            }
        }
        return min_idx;
    }

    void c_smo_solve(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                     SyncData<float_type> &alpha_diff,
                     const SyncData<int> &working_set, float_type Cp, float_type Cn,
                     const SyncData<float_type> &k_mat_rows,
                     const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                     int max_iter) {
        //todo rewrite these codes
        //allocate shared memory

        int ws_size = working_set.size();
        int *shared_mem = new int[ws_size * 3 + 2];
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float *a_old = new float[ws_size];
        float *kIwsI = new float[ws_size];
        float *f = new float[ws_size];
        for (int tid = 0; tid < ws_size; ++tid) {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = alpha[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float local_eps;
        int numOfIter = 0;
        while (1) {
            //select fUp and fLow
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (is_I_up(alpha[wsi], y[wsi], Cp, Cn))
                    f_val2reduce[tid] = f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int i = get_min_idx(f_val2reduce, ws_size);
            float up_value = f_val2reduce[i];
            for (int tid = 0; tid < ws_size; ++tid) {
                kIwsI[tid] = k_mat_rows[row_len * i + working_set[tid]];//K[i, wsi]
            }

            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (is_I_low(alpha[wsi], y[wsi], Cp, Cn))
                    f_val2reduce[tid] = -f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int j1 = get_min_idx(f_val2reduce, ws_size);
            float low_value = -f_val2reduce[j1];

            float local_diff = low_value - up_value;
            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
            }

            if (local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                diff[0] = local_diff;
                break;
            }

            //select j2 using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value > -f[tid] && (is_I_low(alpha[wsi], y[wsi], Cp, Cn))) {
                    float aIJ = kd[i] + kd[tid] - 2 * kIwsI[tid];
                    float bIJ = -up_value + f[tid];
                    f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[tid] = INFINITY;
            }
            int j2 = get_min_idx(f_val2reduce, ws_size);

            //update alpha
//            if (tid == i)
            *alpha_i_diff = y[working_set[i]] > 0 ? Cp - alpha[working_set[i]] : alpha[working_set[i]];
//            if (tid == j2)
            *alpha_j_diff = min(y[working_set[j2]] > 0 ? alpha[working_set[j2]] : Cn - alpha[working_set[j2]],
                                (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            float l = min(*alpha_i_diff, *alpha_j_diff);

//            if (tid == i)
            alpha[working_set[i]] += l * y[working_set[i]];
//            if (tid == j2)
            alpha[working_set[j2]] -= l * y[working_set[j2]];

            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                float kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
            if (numOfIter > max_iter) break;
        }
        delete[] a_old;
        delete[] f;
        delete[] kIwsI;
        delete[] shared_mem;
    }

    void nu_smo_solve(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                      SyncData<float_type> &alpha_diff,
                      const SyncData<int> &working_set, float_type C, const SyncData<float_type> &k_mat_rows,
                      const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                      int max_iter) {
        //allocate shared memory
        int ws_size = working_set.size();
        int *shared_mem = new int[ws_size * 3 + 2];
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float *a_old = new float[ws_size];
        float *kIpwsI = new float[ws_size];
        float *kInwsI = new float[ws_size];
        float *f = new float[ws_size];
        for (int tid = 0; tid < ws_size; ++tid) {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = alpha[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float local_eps;
        int numOfIter = 0;
        while (1) {
            //select I_up (y=+1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] < C)
                    f_val2reduce[tid] = f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int ip = get_min_idx(f_val2reduce, ws_size);
            float up_value_p = f_val2reduce[ip];
            for (int tid = 0; tid < ws_size; ++tid) {
                kIpwsI[tid] = k_mat_rows[row_len * ip + working_set[tid]];//K[i, wsi]
            }

            //select I_up (y=-1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] > 0)
                    f_val2reduce[tid] = f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int in = get_min_idx(f_val2reduce, ws_size);
            float up_value_n = f_val2reduce[in];
            for (int tid = 0; tid < ws_size; ++tid) {
                kInwsI[tid] = k_mat_rows[row_len * in + working_set[tid]];//K[i, wsi]
            }

            //select I_low (y=+1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] > 0)
                    f_val2reduce[tid] = -f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int j1p = get_min_idx(f_val2reduce, ws_size);
            float low_value_p = -f_val2reduce[j1p];

            //select I_low (y=-1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] < C)
                    f_val2reduce[tid] = -f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int j1n = get_min_idx(f_val2reduce, ws_size);
            float low_value_n = -f_val2reduce[j1n];

            float local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);

            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                diff[0] = local_diff;
            }

            if (local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                break;
            }

            //select j2p using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_p > -f[tid] && y[wsi] > 0 && alpha[wsi] > 0) {
                    float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float bIJ = -up_value_p + f[tid];
                    f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[tid] = INFINITY;
            }
            int j2p = get_min_idx(f_val2reduce, ws_size);
            float f_val_j2p = f_val2reduce[j2p];

            //select j2n using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_n > -f[tid] && y[wsi] < 0 && alpha[wsi] < C) {
                    float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float bIJ = -up_value_n + f[tid];
                    f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[tid] = INFINITY;
            }
            int j2n = get_min_idx(f_val2reduce, ws_size);

            int i, j2;
            float up_value;
            float *kIwsI;
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
//            if (tid == i)
//                *alpha_i_diff = y > 0 ? C - a : a;
            *alpha_i_diff = y[working_set[i]] > 0 ? C - alpha[working_set[i]] : alpha[working_set[i]];
//            if (tid == j2)
//                *alpha_j_diff = min(y > 0 ? a : C - a, (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
            *alpha_j_diff = min(y[working_set[j2]] > 0 ? alpha[working_set[j2]] : C - alpha[working_set[j2]],
                                (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
//            __syncthreads();
            float l = min(*alpha_i_diff, *alpha_j_diff);

            alpha[working_set[i]] += l * y[working_set[i]];
            alpha[working_set[j2]] -= l * y[working_set[j2]];

            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                float kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
            if (numOfIter > max_iter) break;
        }
        delete[] a_old;
        delete[] f;
        delete[] kIpwsI;
        delete[] kInwsI;
        delete[] shared_mem;
    }

    void
    update_f(SyncData<float_type> &f, const SyncData<float_type> &alpha_diff, const SyncData<float_type> &k_mat_rows,
             int n_instances) {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        KERNEL_LOOP(idx, n_instances) {//one thread to update multiple fvalues.
            float_type sum_diff = 0;
            for (int i = 0; i < alpha_diff.size(); ++i) {
                float_type d = alpha_diff[i];
                if (d != 0) {
                    sum_diff += d * k_mat_rows[i * n_instances + idx];
                }
            }
            f[idx] -= sum_diff;
        }
    }

    void sort_f(SyncData<float_type> &f_val2sort, SyncData<int> &f_idx2sort) {
        vector<std::pair<float_type, int>> paris;
        for (int i = 0; i < f_val2sort.size(); ++i) {
            paris.emplace_back(f_val2sort[i], f_idx2sort[i]);
        }
        std::sort(paris.begin(), paris.end());
        for (int i = 0; i < f_idx2sort.size(); ++i) {
            f_idx2sort[i] = paris[i].second;
        }
    }
}

