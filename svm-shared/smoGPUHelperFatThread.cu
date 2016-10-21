/**
 * smoGPUHelperFatThread.cu
 * This file contains kernel functions for working set selection
 * Created on: Jun 1, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/
#include "smoGPUHelper.h"
#include "gpu_global_utility.h"
#include "devUtility.h"
#include <float.h>

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
									int nNumofTraingSamples, float_point *pfBlockMin, int *pnBlockMinGlobalKey)
{
	//__shared__ float_point fTempLocalYiFValue[BLOCK_SIZE * TASK_OF_THREAD];
	//__shared__ int nTempLocalKeys[BLOCK_SIZE * TASK_OF_THREAD];
	__shared__ float_point fTempLocalYiFValue[BLOCK_SIZE];
	__shared__ int nTempLocalKeys[BLOCK_SIZE];

	int nThreadId = threadIdx.x;
	fTempLocalYiFValue[nThreadId] = FLT_MAX;
	int nBlockId = blockIdx.y * gridDim.x + blockIdx.x;

	float_point fAlpha;
	//float_point test= FLT_MAX;
	int nLabel;
	int nArrayIndex;

	//#pragma unroll TASK_OF_THREAD
	for(int i = 0; i < TASK_OF_THREAD_UP; i++)
	{
		nArrayIndex = (nBlockId * BLOCK_SIZE * TASK_OF_THREAD_UP)  + i * BLOCK_SIZE + nThreadId;
		//fTempLocalYiFValue[nThreadId + (i * BLOCK_SIZE)] = FLT_MAX;
		if(nArrayIndex < nNumofTraingSamples)
		{
			fAlpha = pfAlpha[nArrayIndex];
			nLabel = pnLabel[nArrayIndex];
			//fill yi*GValue in a block
			if((nLabel > 0 && fAlpha < fPCost) || (nLabel < 0 && fAlpha > 0))
			{
				float_point fTemp = pfYiFValue[nArrayIndex];
				if(fTempLocalYiFValue[nThreadId] > fTemp)
				{
					fTempLocalYiFValue[nThreadId] = fTemp;
					nTempLocalKeys[nThreadId] = nArrayIndex;
				}
			}
		}
	}

	__syncthreads();	//synchronize threads within a block, and start to do reduce
	GetMinValueOriginal(fTempLocalYiFValue, nTempLocalKeys, BLOCK_SIZE);

	if(nThreadId == 0)
	{
		//int nBlockId =  blockIdx.y * gridDim.x + blockIdx.x;
		pfBlockMin[nBlockId] = fTempLocalYiFValue[0];
		//if(fTempLocalYiFValue[0] == pfYiFValue[18648])
		//	printf("good, %f, key %d %d\n", fTempLocalYiFValue[0], nTempLocalKeys[0], nBlockId);
		pnBlockMinGlobalKey[nBlockId] = nTempLocalKeys[0];
	}
}

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
									int *pnBlockMinGlobalKey, float_point *pfBlockMinYiFValue)
{
	//__shared__ int nTempKey[BLOCK_SIZE * TASK_OF_THREAD];
	//__shared__ float_point fTempMinValues[BLOCK_SIZE * TASK_OF_THREAD];
	//__shared__ float_point fTempObjValues[BLOCK_SIZE * TASK_OF_THREAD];
	__shared__ int nTempKey[BLOCK_SIZE];
	__shared__ float_point fTempMinValues[BLOCK_SIZE];
	__shared__ float_point fTempObjValues[BLOCK_SIZE];

	int nThreadId = threadIdx.x;

	fTempMinValues[nThreadId] = FLT_MAX;
	fTempObjValues[nThreadId] = FLT_MAX;
	int nBlockId = blockIdx.y * gridDim.x + blockIdx.x;

	//fill data (-b_ij * b_ij/a_ij) into a block
	float_point fYiGValue;
	float_point fBeta;

	float_point fAlpha;
	int nLabel;
	int nArrayIndex;

	//#pragma unroll TASK_OF_THREAD
	float_point fUpValue;
	float_point fBUp_j;
	float_point fAUp_j;
	for(int i = 0; i < TASK_OF_THREAD_UP; i++)
	{
		nArrayIndex = (nBlockId * BLOCK_SIZE * TASK_OF_THREAD_UP)  + i * BLOCK_SIZE + nThreadId;
		//fTempMinValues[nThreadId + (i * BLOCK_SIZE)] = FLT_MAX;
		//fTempObjValues[nThreadId + (i * BLOCK_SIZE)] = FLT_MAX;
		if(nArrayIndex < nNumofTrainingSamples)
		{
			fUpValue = fMinusYiUpValue;
			fYiGValue = pfYiFValue[nArrayIndex];
			fAlpha = pfAlpha[nArrayIndex];

			nLabel = pnLabel[nArrayIndex];
			//nTempKey[nThreadId + (i * BLOCK_SIZE)] = nArrayIndex;

			/*************** calculate b_ij ****************/
			//b_ij = -Gi + Gj in paper, but b_ij = -Gi + y_j * Gj in the code of libsvm. Here we follow the code of libsvm
			fBUp_j = fUpValue + fYiGValue;

		    if(((nLabel > 0) && (fAlpha > 0)) ||
		       ((nLabel < 0) && (fAlpha < fNCost))
		    	)
			{
		    	int nExact = nArrayIndex;
		    	if(nExact >= nNumofInstance)
		    		nExact -= nNumofInstance;
				fAUp_j = 2 - 2 * pfHessianRow[nExact];

				if(fAUp_j <= 0)
				{
					fAUp_j = TAU;
				}

			    if(fBUp_j > 0)
			    {
			    	//for getting optimized pair
					fBeta = -(__fdividef(__powf(fBUp_j, 2.0f), fAUp_j));
			    	//fTempMinYiFValue[nThreadId] = -fYiGValue;
					//fTempObjValues[nThreadId + (i * BLOCK_SIZE)] = fBeta;

					if(fTempObjValues[nThreadId] > fBeta)
					{
						fTempObjValues[nThreadId] = fBeta;
						nTempKey[nThreadId] = nArrayIndex;
					}
			    }
			    //fTempMinValues[nThreadId + (i * BLOCK_SIZE)] = -fYiGValue;
			    if(fTempMinValues[nThreadId] > -fYiGValue)
			    {
			    	fTempMinValues[nThreadId] = -fYiGValue;
			    }
			}
		}
	}


	__syncthreads();
	GetMinValueOriginal(fTempMinValues, BLOCK_SIZE);
	if(nThreadId == 0)
	{
		pfBlockMinYiFValue[nBlockId] = fTempMinValues[0];
	}

	//block level reduce
	__syncthreads();
	GetMinValueOriginal(fTempObjValues, nTempKey, BLOCK_SIZE);


	if(nThreadId == 0)
	{
		pfBlockMin[nBlockId] = fTempObjValues[0];
		pnBlockMinGlobalKey[nBlockId] = nTempKey[0];
	}
}


/*
 * @brief: use reducer to get the minimun value in parallel
 * @param: pfValues: a pointer to a set of data
 * @param: pnKey:	 a pointer to the index of the set of data. It's for getting the location of min.
 */
__device__ void GetBigMinValue(float_point *pfValues, int *pnKey)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;
	float_point fValue1, fValue2;
	fValue1 = pfValues[nTid];
	int nNumofBlock = BLOCK_SIZE * TASK_OF_THREAD;

	for(int i = BLOCK_SIZE * (TASK_OF_THREAD - 1); i >= BLOCK_SIZE; i -= BLOCK_SIZE)
	{
		compOffset = nTid + i;
		if(nTid < i && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}

	GetMinValue(pfValues, pnKey, BLOCK_SIZE);
	/*#pragma unroll UNROLL_REDUCE
	for(int i = (BLOCK_SIZE / 2); i >= 1; i /= 2)
	{
		compOffset = nTid + i;
		if(nTid < i)
		{
			if(compOffset < nNumofBlock)
			{
				fValue2 = pfValues[compOffset];
				if(fValue2 < fValue1)
				{
					pnKey[nTid] = pnKey[compOffset];
					pfValues[nTid] = fValue2;
					fValue1 = fValue2;
				}
			}
		}
		else
			return;
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}*/

}

__device__ void GetBigMinValue(float_point *pfValues)
{
	int nTid = threadIdx.x;
	int compOffset;
	float_point fValue1, fValue2;
	fValue1 = pfValues[nTid];
	int nNumofBlock = BLOCK_SIZE * TASK_OF_THREAD;

	for(int i = BLOCK_SIZE * (TASK_OF_THREAD - 1); i >= BLOCK_SIZE; i -= BLOCK_SIZE)
	{
		compOffset = nTid + i;
		if(nTid < i && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}

	GetMinValue(pfValues, BLOCK_SIZE);
}
