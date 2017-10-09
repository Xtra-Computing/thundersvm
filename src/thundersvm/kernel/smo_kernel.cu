//
// Created by jiashuai on 17-9-21.
//
#include <cfloat>
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
localSMO(const int *label, real *FValues, real *alpha, real *alpha_diff, const int *working_set, int ws_size, float C,
         const float *k_mat_rows, int row_len, real eps, real *diff_and_bias) {
	//"row_len" equals to the number of instances in the original training dataset.
    //allocate shared memory
    extern __shared__ int sharedMem[];
    int *idx4Reduce = sharedMem; //temporary memory for reduction
    float *fValuesI = (float *) &idx4Reduce[ws_size]; //fvalues of I_up.
    float *fValuesJ = &fValuesI[ws_size]; //fvalues of I_low
    float *alphaIDiff = &fValuesJ[ws_size]; //delta alpha_i
    float *alphaJDiff = &alphaIDiff[1];

    //index, f value and alpha for each instance
    int tid = threadIdx.x;
    int wsi = working_set[tid];
    float y = label[wsi];
    float f = FValues[wsi];
    float a = alpha[wsi];
    float aold = a;
    __syncthreads();
    float local_eps;
    int numOfIter = 0;
    while (1) {
        //select fUp and fLow
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            fValuesI[tid] = f;
        else
            fValuesI[tid] = FLT_MAX;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            fValuesJ[tid] = -f;
        else
            fValuesJ[tid] = FLT_MAX;
        int i = get_block_min(fValuesI, idx4Reduce);
        float upValue = fValuesI[i];
        float kIwsI = k_mat_rows[row_len * i + wsi];//K[i, wsi]
        __syncthreads();
        int j1 = get_block_min(fValuesJ, idx4Reduce);
        float lowValue = -fValuesJ[j1];

        float local_diff = lowValue - upValue;
        if (numOfIter == 0) {
            local_eps = max(eps, 0.1f * local_diff);
        }

        if (local_diff < local_eps) {
            alpha[wsi] = a;
            alpha_diff[tid] = -(a - aold) * y;
            if (tid == 0) {
                diff_and_bias[0] = local_diff;
                diff_and_bias[1] = (lowValue + upValue) / 2;
            }
            break;
        }
        __syncthreads();

        //select j2 using second order heuristic
        if (-upValue > -f && ((y > 0 && a > 0) || (y < 0 && a < C))) {
            float aIJ = 1 + 1 - 2 * kIwsI;
            float bIJ = -upValue + f;
            fValuesI[tid] = -bIJ * bIJ / aIJ;
        } else
            fValuesI[tid] = FLT_MAX;
        int j2 = get_block_min(fValuesI, idx4Reduce);

        //update alpha
        if (tid == i)
            *alphaIDiff = y > 0 ? C - a : a;
        if (tid == j2)
            *alphaJDiff = min(y > 0 ? a : C - a, (-upValue - fValuesJ[j2]) / (1 + 1 - 2 * kIwsI));
        __syncthreads();
        float l = min(*alphaIDiff, *alphaJDiff);

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
        real sumDiff = 0;
        for (int i = 0; i < ws_size; ++i) {
            real d = alpha_diff[i];
            if (d != 0)
                sumDiff += d * k_mat_rows[i * n_instances + idx];
        }
        f[idx] -= sumDiff;
    }
}
