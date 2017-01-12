/*
 * predictionGPUHelper.h
 *
 *  Created on: 1 Jan 2017
 *      Author: Zeyi Wen
 */

#ifndef MASCOT_PREDICTIONGPUHELPER_H_
#define MASCOT_PREDICTIONGPUHELPER_H_
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "../SharedUtility/DataType.h"

__global__ void rbfKernel(const float_point *sampleSelfDot, int numOfSamples,
                          const float_point *svMapSelfDot, int svMapSize,
                          float_point *kernelValues, float_point gamma);

__global__ void sumKernelValues(const float_point *kernelValues, int numOfSamples, int svMapSize, int cnr2,
                                const int *svIndex, const float_point *coef,
                                const int *start, const int *count,
                                const float_point *bias, float_point *decValues);


#endif /* MASCOT_PREDICTIONGPUHELPER_H_ */
