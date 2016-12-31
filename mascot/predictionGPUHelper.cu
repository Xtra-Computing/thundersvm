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
__global__ void rbfKernel(const float_point *sampleSelfDot, int numOfSamples,
                          const float_point *svMapSelfDot, int svMapSize,
                          float_point *kernelValues, float_point gamma) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / svMapSize;
    int SVId = idx % svMapSize;
    if (sampleId < numOfSamples) {
        float_point sampleDot = sampleSelfDot[sampleId];
        float_point svDot = svMapSelfDot[SVId];
        float_point dot = kernelValues[idx];			//dot product of two instances
        kernelValues[idx] = expf(-gamma * (sampleDot + svDot - 2 * dot));
    }
};

/**
 * @brief: compute decision values, given kernel values and model info.
 */
__global__ void sumKernelValues(const float_point *kernelValues, int numOfSamples, int svMapSize, int cnr2,
                                const int *svIndex, const float_point *coef,
                                const int *start, const int *count,
                                const float_point *bias, float_point *decValues) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / cnr2;
    int modelId = idx % cnr2;
    if (sampleId < numOfSamples) {
        float_point sum = 0;
        const float_point *kernelValue = kernelValues + sampleId * svMapSize;
        int si = start[modelId];
        int ci = count[modelId];
        for (int i = 0; i < ci; ++i) {//can be improved.
            sum += coef[si + i] * kernelValue[svIndex[si + i]];
        }
        decValues[idx] = sum - bias[modelId];
    }
}


