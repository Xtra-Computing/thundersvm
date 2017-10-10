//
// Created by jiashuai on 17-9-21.
//
#include "thundersvm/kernel/smo_kernel.h"

__device__ int get_block_min(const float *values, int *index) {
    int tid = threadIdx.x;
    index[tid] = tid;
    __syncthreads();
    //block size is always the power of 2
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (values[index[tid + offset]] < values[index[tid]]) {
                index[tid] = index[tid + offset];
            }
        }
        __syncthreads();
    }
    return index[0];
}

__global__ void
local_smo(const int *label, real *f_values, real *alpha, real *alpha_diff, const int *working_set, int ws_size, float C,
          const float *k_mat_rows, int row_len, real eps, real *diff_and_bias) {
    //"row_len" equals to the number of instances in the original training dataset.
    //allocate shared memory
    extern __shared__ int shared_mem[];
    int *idx2reduce = shared_mem; //temporary memory for reduction
    float *f_values_i = (float *) &idx2reduce[ws_size]; //fvalues of I_up.
    float *f_values_j = &f_values_i[ws_size]; //fvalues of I_low
    float *alpha_i_diff = &f_values_j[ws_size]; //delta alpha_i
    float *alpha_j_diff = &alpha_i_diff[1];

    //index, f value and alpha for each instance
    int tid = threadIdx.x;
    int wsi = working_set[tid];
    float y = label[wsi];
    float f = f_values[wsi];
    float a = alpha[wsi];
    float aold = a;
    __syncthreads();
    float local_eps;
    int numOfIter = 0;
    while (1) {
        //select fUp and fLow
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            f_values_i[tid] = f;
        else
            f_values_i[tid] = INFINITY;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            f_values_j[tid] = -f;
        else
            f_values_j[tid] = INFINITY;
        int i = get_block_min(f_values_i, idx2reduce);
        float up_value = f_values_i[i];
        float kIwsI = k_mat_rows[row_len * i + wsi];//K[i, wsi]
        __syncthreads();
        int j1 = get_block_min(f_values_j, idx2reduce);
        float low_value = -f_values_j[j1];

        float local_diff = low_value - up_value;
        if (numOfIter == 0) {
            local_eps = max(eps, 0.1f * local_diff);
        }

        if (local_diff < local_eps) {
            alpha[wsi] = a;
            alpha_diff[tid] = -(a - aold) * y;
            if (tid == 0) {
                diff_and_bias[0] = local_diff;
                diff_and_bias[1] = (low_value + up_value) / 2;
            }
            break;
        }
        __syncthreads();

        //select j2 using second order heuristic
        if (-up_value > -f && ((y > 0 && a > 0) || (y < 0 && a < C))) {
            float aIJ = 1 + 1 - 2 * kIwsI;
            float bIJ = -up_value + f;
            f_values_i[tid] = -bIJ * bIJ / aIJ;
        } else
            f_values_i[tid] = INFINITY;
        int j2 = get_block_min(f_values_i, idx2reduce);

        //update alpha
        if (tid == i)
            *alpha_i_diff = y > 0 ? C - a : a;
        if (tid == j2)
            *alpha_j_diff = min(y > 0 ? a : C - a, (-up_value - f_values_j[j2]) / (1 + 1 - 2 * kIwsI));
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
    }
}

__global__ void update_f(real *f, int ws_size, const real *alpha_diff, const real *k_mat_rows, int n_instances) {
    //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
    KERNEL_LOOP(idx, n_instances) {//one thread to update multiple fvalues.
        real sum_diff = 0;
        for (int i = 0; i < ws_size; ++i) {
            real d = alpha_diff[i];
            if (d != 0)
                sum_diff += d * k_mat_rows[i * n_instances + idx];
        }
        f[idx] -= sum_diff;
    }
}
