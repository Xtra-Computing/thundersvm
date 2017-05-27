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

__global__ void rbfKernel(const real *sampleSelfDot, int numOfSamples,
                          const real *svMapSelfDot, int svMapSize,
                          real *kernelValues, real gamma);

__global__ void sumKernelValues(const real *kernelValues, int numOfSamples, int svMapSize, int cnr2,
                                const int *svIndex, const real *coef,
                                const int *start, const int *count,
                                const real *bias, real *decValues);


#endif /* MASCOT_PREDICTIONGPUHELPER_H_ */
