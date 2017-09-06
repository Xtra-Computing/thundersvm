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
__global__ void GetBlockMinYiGValue(real *pfYiFValue, real *pfAlpha, int *pnLabel, real fPCost,
									int nNumofTraingSamples, real *pfBlockMin, int *pnBlockMinGlobalKey);
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
__global__ void GetBlockMinLowValue(real *pfYiFValue, real *pfAlpha, int *pnLabel, real fNCost,
									int nNumofTrainingSamples, real *pfDiagHessian, real *pfHessianRow,
									real fMinusYiUpValue, real fUpValueKernel, real *pfBlockMin,
									int *pnBlockMinGlobalKey, real *pfBlockMinYiFValue);

/*
 * @brief: kernel function for getting the minimum value in a set of block min values
 * @param: pfBlockMin: a set of min value returned from block level reducer
 * @param: pnBlockMinKey: a set of indices for block min (i.e., each block min value has a global index)
 * @param: nNumofBlock:	  the number of blocks
 * @param: pfMinValue:	  a pointer to global min value (the result of this function)
 * @param: pnMinKey:	  a pointer to the index of the global min value (the result of this function)
 */
__global__ void GetGlobalMin(real *pfBlockMin, int *pnBlockMinKey, int nNumofBlock,
							 real *pfYiFValue, real *pfHessianRow, real *pfTempKeyValue);
__global__ void GetGlobalMin(real *pfBlockMin, int nNumofBlock, real *pfTempKeyValue);
/*
 * @brief: update gradient values for all samples
 * @param: pfYiFValue: the gradient of samples (input and output of this kernel)
 * @param: pfHessianRow1: the Hessian row of sample one
 * @param: pfHessianRow2: the Hessian row of sample two
 * @param: fY1AlphaDiff: the difference of old and new alpha of sample one
 * @param: fY2AlphaDiff: the difference of old and new alpha of sample two
 */
__global__ void UpdateYiFValueKernel(real *pfAlpha, real *pDevBuffer, real *pfYiFValue,
									 real *pfHessianRow1, real *pfHessianRow2,
							    	 real fY1AlphaDiff, real fY2AlphaDiff, int nNumofTrainingSamples);

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
extern __device__ real devDiff;//fUp - fLow
extern __device__ real devRho;//bias
/**
 * local SMO in one block
 * @param label: label for all instances in training set
 * @param FValues: f value for all instances in training set
 * @param alpha: alpha for all instances in training set
 * @param alphaDiff: difference of each alpha in working set after local SMO
 * @param workingSet: index of each instance in training set
 * @param wsSize: size of working set
 * @param C: C parameter in SVM
 * @param hessianMatrixCache: |working set| * |training set| kernel matrix, row major
 * @param ld: number of instances each row in hessianMatrixCache
 */

__global__ void localSMO(const int *label, real *FValues, real *alpha, real *alphaDiff,
						 const int *workingSet, int wsSize, float C, const float *hessianMatrixCache, int ld);
/**
 * update f values using alpha diff
 * @param FValues: f values for all instances in training set
 * @param label: label for all instances in training set
 * @param workingSet: index of each instance in working set
 * @param wsSize: size of working set
 * @param alphaDiff: difference of alpha in working set
 * @param hessianMatrixCache: |working set| * |training set| kernel matrix, row major
 * @param numOfSamples
 */
__global__ void updateF(real *FValues, const int *label, const int *workingSet, int wsSize, const real *alphaDiff,
		const real *hessianMatrixCache, int numOfSamples);
__global__ void getFUpValues(const real *FValues, const real *alpha, const int *labels,
                             int numOfSamples, real C, real *FValue4Sort, int *Idx4Sort, const int *);
__global__ void getFLowValues(const real *FValues, const real *alpha, const int *labels,
                              int numOfSamples, real C, real *FValue4Sort, int *Idx4Sort, const int *);
#endif /* WORKINGSETGPUHELPER_H_ */
