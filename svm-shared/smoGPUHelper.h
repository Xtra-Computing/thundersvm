/**
 * smoGPUHelper.h
 * Created on: May 29, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef WORKINGSETGPUHELPER_H_
#define WORKINGSETGPUHELPER_H_

//GPU header files
#include <helper_cuda.h>

#include "constant.h"
#include "gpu_global_utility.h"

//device function for CPairSelector

/*
 * @brief: kernel funciton for getting minimum value within a block
 * @param: pfYiFValue: a set of value = y_i * gradient of subjective function
 * @param: pfAlpha:	   a set of alpha related to training samples
 * @param: pnLabel:	   a set of labels related to training samples
 * @param: nNumofTrainingSamples: the number of training samples
 * @param: pfBlockMin: the min value of this block (function result)
 * @param: pnBlockMinGlobalKey: the index of the min value of this block
 */
__global__ void GetBlockMinYiGValue(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fPCost,
									int nNumofTraingSamples, float_point *pfBlockMin, int *pnBlockMinGlobalKey);
/*
 * @brief: for selecting the second sample to optimize
 * @param: pfYiFValue: the gradient of data samples
 * @param: pfAlpha: alpha values for samples
 * @param: fNCost: the cost of negative sample (i.e., the C in SVM)
 * @param: pfDiagHessian: the diagonal of Hessian Matrix
 * @param: pfHessianRow: a Hessian row of sample one
 * @param: fMinusYiUpValue: -yi*gradient of sample one
 * @param: fUpValueKernel: self dot product of sample one
 * @param: pfBlockMin: minimum value of each block (the output of this kernel)
 * @param: pnBlockMinGlobalKey: the key of each block minimum value (the output of this kernel)
 * @param: pfBlockMinYiFValue: the block minimum gradient (the output of this kernel. for convergence check)
 */
__global__ void GetBlockMinLowValue(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fNCost,
									int nNumofTrainingSamples, float_point *pfDiagHessian, float_point *pfHessianRow,
									float_point fMinusYiUpValue, float_point fUpValueKernel, float_point *pfBlockMin,
									int *pnBlockMinGlobalKey, float_point *pfBlockMinYiFValue);

/*
 * @brief: kernel function for getting the minimum value in a set of block min values
 * @param: pfBlockMin: a set of min value returned from block level reducer
 * @param: pnBlockMinKey: a set of indices for block min (i.e., each block min value has a global index)
 * @param: nNumofBlock:	  the number of blocks
 * @param: pfMinValue:	  a pointer to global min value (the result of this function)
 * @param: pnMinKey:	  a pointer to the index of the global min value (the result of this function)
 */
__global__ void GetGlobalMin(float_point *pfBlockMin, int *pnBlockMinKey, int nNumofBlock,
							 float_point *pfYiFValue, float_point *pfHessianRow, float_point *pfTempKeyValue);
__global__ void GetGlobalMin(float_point *pfBlockMin, int nNumofBlock, float_point *pfTempKeyValue);
/*
 * @brief: update gradient values for all samples
 * @param: pfYiFValue: the gradient of samples (input and output of this kernel)
 * @param: pfHessianRow1: the Hessian row of sample one
 * @param: pfHessianRow2: the Hessian row of sample two
 * @param: fY1AlphaDiff: the difference of old and new alpha of sample one
 * @param: fY2AlphaDiff: the difference of old and new alpha of sample two
 */
__global__ void UpdateYiFValueKernel(float_point *pfAlpha, float_point *pDevBuffer, float_point *pfYiFValue,
									 float_point *pfHessianRow1, float_point *pfHessianRow2,
							    	 float_point fY1AlphaDiff, float_point fY2AlphaDiff, int nNumofTrainingSamples);


/*
 * @brief: kernel funciton for getting minimum value within a block
 * @param: pfYiFValue: a set of value = y_i * gradient of subjective function
 * @param: pfAlpha:	   a set of alpha related to training samples
 * @param: pnLabel:	   a set of labels related to training samples
 * @param: nNumofTrainingSamples: the number of training samples
 * @param: pfBlockMin: the min value of this block (function result)
 * @param: pnBlockMinGlobalKey: the index of the min value of this block
 */
__global__ void GetBigBlockMinYiGValue(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fPCost,
									int nNumofTraingSamples, float_point *pfBlockMin, int *pnBlockMinGlobalKey);

/*
 * @brief: for selecting the second sample to optimize
 * @param: pfYiFValue: the gradient of data samples
 * @param: pfAlpha: alpha values for samples
 * @param: fNCost: the cost of negative sample (i.e., the C in SVM)
 * @param: pfDiagHessian: the diagonal of Hessian Matrix
 * @param: pfHessianRow: a Hessian row of sample one
 * @param: fMinusYiUpValue: -yi*gradient of sample one
 * @param: fUpValueKernel: self dot product of sample one
 * @param: pfBlockMin: minimum value of each block (the output of this kernel)
 * @param: pnBlockMinGlobalKey: the key of each block minimum value (the output of this kernel)
 * @param: pfBlockMinYiFValue: the block minimum gradient (the output of this kernel. for convergence check)
 */
__global__ void GetBigBlockMinLowValue(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fNCost,
									int nNumofTrainingSamples, int nNumofInstance, float_point *pfDiagHessian, float_point *pfHessianRow,
									float_point fMinusYiUpValue, float_point fUpValueKernel, float_point *pfBlockMin,
									int *pnBlockMinGlobalKey, float_point *pfBlockMinYiFValue);

#endif /* WORKINGSETGPUHELPER_H_ */
