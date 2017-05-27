/*
 * predictionGPUHelper.cu
 *
 *  Created on: 1 Jan 2017
 *      Author: Zeyi Wen
 */

#include "predictionGPUHelper.h"

/**
 * @brief: compute the RBF kernel values, given dot products.
 */
__global__ void rbfKernel(const real *sampleSelfDot, int numOfSamples,
                          const real *svMapSelfDot, int svMapSize,
                          real *kernelValues, real gamma) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / svMapSize;
    int SVId = idx % svMapSize;
    if (sampleId < numOfSamples) {
        real sampleDot = sampleSelfDot[sampleId];
        real svDot = svMapSelfDot[SVId];
        real dot = kernelValues[idx];			//dot product of two instances
        kernelValues[idx] = expf(-gamma * (sampleDot + svDot - 2 * dot));
    }
};

/**
 * @brief: compute decision values, given kernel values and model info.
 */
__global__ void sumKernelValues(const real *kernelValues, int numOfSamples, int svMapSize, int cnr2,
                                const int *svIndex, const real *coef,
                                const int *start, const int *count,
                                const real *bias, real *decValues) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / cnr2;
    int modelId = idx % cnr2;
    if (sampleId < numOfSamples) {
        real sum = 0;
        const real *kernelValue = kernelValues + sampleId * svMapSize;//kernel values of this instance
        int si = start[modelId];
        int ci = count[modelId];
        for (int i = 0; i < ci; ++i) {//can be improved.
            sum += coef[si + i] * kernelValue[svIndex[si + i]];
        }
        decValues[idx] = sum - bias[modelId];
    }
}


