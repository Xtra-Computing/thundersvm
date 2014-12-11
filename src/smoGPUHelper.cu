
#include "smoGPUHelper.h"
//#include "devUtility2.h"
#include <float.h>

//-----------------------------------

__device__ void GetMinValueOriginal(float_point *pfValues, int *pnKey, int nNumofBlock)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64)
		{
			if(compOffset < nNumofBlock)
			{
				if(pfValues[compOffset] < pfValues[nTid])
				{
					pnKey[nTid] = pnKey[compOffset];
					pfValues[nTid] = pfValues[compOffset];
				}
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

}

__device__ void GetMinValueOriginal(float_point *pfValues, int nNumofBlock)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
//		else
//			return;
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
//		else
//			return;
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
//		else
//			return;

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
//		else
//			return;

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
//		else
//			return;


		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
//		else
//			return;

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}

}


/* *
 /*
 * @brief: use reducer to get the minimun value in parallel
 * @param: pfValues: a pointer to a set of data
 * @param: pnKey:	 a pointer to the index of the set of data. It's for getting the location of min.
 */
__device__ void GetMinValue(float_point *pfValues, int *pnKey, int nNumofBlock)
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

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64)
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
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
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

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				//fValue1 = fValue2;
			}
		}

}

__device__ void GetMinValue(float_point *pfValues, int nNumofBlock)
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

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
//		else if(nTid >= 64)
//			return;
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
//		else if(nTid >= 32)
//			return;
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
//		else
//			return;

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
//		else
//			return;

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
//		else
//			return;


		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
//		else
//			return;

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			pfValues[nTid] = (fValue2 < fValue1) ? fValue2 : fValue1;
		}

}


/* *
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

//-----------------------------------


/* *
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
									int nNumofTraingSamples, float_point *pfBlockMin, int *pnBlockMinGlobalKey)
{
	__shared__ float_point fTempLocalYiFValue[BLOCK_SIZE];
	__shared__ int nTempLocalKeys[BLOCK_SIZE];

	int nGlobalIndex;
	int nThreadId = threadIdx.x;
	nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread

	float_point fAlpha;
	int nLabel;
	fAlpha = pfAlpha[nGlobalIndex];
	nLabel = pnLabel[nGlobalIndex];
	fTempLocalYiFValue[nThreadId] = FLT_MAX;
	//fill yi*GValue in a block
	if(nGlobalIndex < nNumofTraingSamples && ((nLabel > 0 && fAlpha < fPCost) || (nLabel < 0 && fAlpha > 0)))
	{
		//I_0 is (fAlpha > 0 && fAlpha < fCostP). This condition is covered by the following condition
		//index set I_up
		fTempLocalYiFValue[nThreadId] = pfYiFValue[nGlobalIndex];
		nTempLocalKeys[nThreadId] = nGlobalIndex;
	}
	__syncthreads();	//synchronize threads within a block, and start to do reduce

	GetMinValueOriginal(fTempLocalYiFValue, nTempLocalKeys, blockDim.x);

	if(nThreadId == 0)
	{
		int nBlockId =  blockIdx.y * gridDim.x + blockIdx.x;
		pfBlockMin[nBlockId] = fTempLocalYiFValue[0];
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
__global__ void GetBlockMinLowValue(float_point *pfYiFValue, float_point *pfAlpha, int *pnLabel, float_point fNCost,
									int nNumofTrainingSamples, float_point *pfDiagHessian, float_point *pfHessianRow,
									float_point fMinusYiUpValue, float_point fUpValueKernel, float_point *pfBlockMin,
									int *pnBlockMinGlobalKey, float_point *pfBlockMinYiFValue)
{
	__shared__ int nTempKey[BLOCK_SIZE];
	__shared__ float_point fTempMinValues[BLOCK_SIZE];

	int nThreadId = threadIdx.x;
	int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread

	fTempMinValues[nThreadId] = FLT_MAX;
	//fTempMinYiFValue[nThreadId] = FLT_MAX;

	//fill data (-b_ij * b_ij/a_ij) into a block
	float_point fYiGValue;
	float_point fBeta;
	int nReduce = NOREDUCE;
	if(nGlobalIndex < nNumofTrainingSamples)
	{
		float_point fUpValue = fMinusYiUpValue;
		fYiGValue = pfYiFValue[nGlobalIndex];
		float_point fAlpha = pfAlpha[nGlobalIndex];

		int nLabel = pnLabel[nGlobalIndex];

		nTempKey[nThreadId] = nGlobalIndex;
		float_point fBUp_j;

		/*************** calculate b_ij ****************/
		//b_ij = -Gi + Gj in paper, but b_ij = -Gi + y_j * Gj in the code of libsvm. Here we follow the code of libsvm
		fBUp_j = fUpValue + fYiGValue;

	    if(((nLabel > 0) && (fAlpha > 0)) ||
	       ((nLabel < 0) && (fAlpha < fNCost))
	    	)
		{
	    	float_point fAUp_j;
			fAUp_j = fUpValueKernel + pfDiagHessian[nGlobalIndex] - 2 * pfHessianRow[nGlobalIndex];

			if(fAUp_j <= 0)
			{
				fAUp_j = TAU;
			}

		    if(fBUp_j > 0)
		    {
		    	nReduce = REDUCE1 | REDUCE0;
		    }
		    else
		    	nReduce = REDUCE0;

	    	//for getting optimized pair
			fBeta = -(fBUp_j * fBUp_j / fAUp_j);
	    	//fTempMinYiFValue[nThreadId] = -fYiGValue;
		}
	}

	if((nReduce & REDUCE0) != 0)
	{
		fTempMinValues[threadIdx.x] = -fYiGValue;
	}
	__syncthreads();
	GetMinValueOriginal(fTempMinValues, blockDim.x);
	int nBlockId;
	if(nThreadId == 0)
	{
		nBlockId =  blockIdx.y * gridDim.x + blockIdx.x;
		pfBlockMinYiFValue[nBlockId] = fTempMinValues[0];
	}

	fTempMinValues[threadIdx.x] = ((nReduce & REDUCE1) != 0) ? fBeta : FLT_MAX;

	//block level reduce
	__syncthreads();
	GetMinValueOriginal(fTempMinValues, nTempKey, blockDim.x);


	if(nThreadId == 0)
	{
		pfBlockMin[nBlockId] = fTempMinValues[0];
		pnBlockMinGlobalKey[nBlockId] = nTempKey[0];
	}
}

/*
 * @brief: kernel function for getting the minimum value in a set of block min values
 * @param: pfBlockMin: a set of min value returned from block level reducer
 * @param: pnBlockMinKey: a set of indices for block min (i.e., each block min value has a global index)
 * @param: nNumofBlock:	  the number of blocks
 * @param: pfMinValue:	  a pointer to global min value (the result of this function)
 * @param: pnMinKey:	  a pointer to the index of the global min value (the result of this function)
 */
__global__ void GetGlobalMin(float_point *pfBlockMin, int *pnBlockMinKey, int nNumofBlock,
							 float_point *pfYiFValue, float_point *pfHessianRow, float_point *pfTempKeyValue)
{
	__shared__ int nTempKey[BLOCK_SIZE];
	__shared__ float_point pfTempMin[BLOCK_SIZE];
	int nThreadId = threadIdx.x;

	if(nThreadId < nNumofBlock)
	{
		nTempKey[nThreadId] = pnBlockMinKey[nThreadId];
		pfTempMin[nThreadId] = pfBlockMin[nThreadId];
	}
	else
	{
		//nTempKey[nThreadId] = pnBlockMinKey[nThreadId];
		pfTempMin[nThreadId] = FLT_MAX;
	}
	//if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
	if(nNumofBlock > BLOCK_SIZE)
	{
		float_point fTempMin = pfTempMin[nThreadId];
		int nTempMinKey = nTempKey[nThreadId];
		for(int i = nThreadId + BLOCK_SIZE; i < nNumofBlock; i += blockDim.x)
		{
			float_point fTempBlockMin = pfBlockMin[i];
			if(fTempBlockMin < fTempMin)
			{
			//store the minimum value and the corresponding key
				fTempMin = fTempBlockMin;
				nTempMinKey = pnBlockMinKey[i];
			}
		}
		nTempKey[nThreadId] = nTempMinKey;
		pfTempMin[nThreadId] = fTempMin;
	}
	 __syncthreads();	//wait until the thread within the block

	 GetMinValue(pfTempMin, nTempKey, nNumofBlock);

	 if(nThreadId == 0)
	 {
		 *(pfTempKeyValue) = (float_point)nTempKey[0];
		 if(pfYiFValue != NULL)
			 *(pfTempKeyValue + 1) = pfYiFValue[nTempKey[0]];//pfTempMin[0];
		 else
			 *(pfTempKeyValue + 1) = pfTempMin[0];

		 if(pfHessianRow != NULL)
			 *(pfTempKeyValue + 2) = pfHessianRow[nTempKey[0]];
	 }
}

/*
 * @brief: kernel function for getting the minimum value in a set of block min values
 * @param: pfBlockMin: a set of min value returned from block level reducer
 * @param: pnBlockMinKey: a set of indices for block min (i.e., each block min value has a global index)
 * @param: nNumofBlock:	  the number of blocks
 * @param: pfMinValue:	  a pointer to global min value (the result of this function)
 * @param: pnMinKey:	  a pointer to the index of the global min value (the result of this function)
 */
__global__ void GetGlobalMin(float_point *pfBlockMin, int nNumofBlock, float_point *pfTempKeyValue)
{
	__shared__ float_point pfTempMin[BLOCK_SIZE];
	int nThreadId = threadIdx.x;

	if(nThreadId < nNumofBlock)
	{
		pfTempMin[nThreadId] = pfBlockMin[nThreadId];
	}

	//if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
	if(nNumofBlock > BLOCK_SIZE)
	{
		float_point fTempMin = pfTempMin[nThreadId];
		for(int i = nThreadId + BLOCK_SIZE; i < nNumofBlock; i += blockDim.x)
		{
			float_point fTempBlockMin = pfBlockMin[i];
			fTempMin = (fTempBlockMin < fTempMin) ? fTempBlockMin : fTempMin;
		}
		pfTempMin[nThreadId] = fTempMin;
	}
	 __syncthreads();	//wait until the thread within the block

	 GetMinValue(pfTempMin, nNumofBlock);

	 if(nThreadId == 0)
	 {
		*(pfTempKeyValue + 3) = pfTempMin[0];
	 }
}

/*
 * @brief: update gradient values for all samples
 * @param: pfYiFValue: the gradient of samples (input and output of this kernel)
 * @param: pfHessianRow1: the Hessian row of sample one
 * @param: pfHessianRow2: the Hessian row of sample two
 * @param: fY1AlphaDiff: the difference of old and new alpha of sample one
 * @param: fY2AlphaDiff: the difference of old and new alpha of sample two
 */
__global__ void UpdateYiFValueKernel(float_point *pfAlpha, float_point *pDevBuffer, float_point *pfYiFValue, float_point *pfHessianRow1, float_point *pfHessianRow2,
							   float_point fY1AlphaDiff, float_point fY2AlphaDiff, int nNumofTrainingSamples)
{
	if(threadIdx.x < 2)
	{
		int nTemp = int(pDevBuffer[threadIdx.x * 2]);
		pfAlpha[nTemp] = pDevBuffer[threadIdx.x * 2 + 1];
		//nTemp = int(pDevBuffer[2]);
		//pfAlpha[nTemp] = pDevBuffer[3];
	}
	__syncthreads();
	float_point fsY1AlphaDiff;
	fsY1AlphaDiff = fY1AlphaDiff;
	float_point fsY2AlphaDiff;
	fsY2AlphaDiff = fY2AlphaDiff;

	int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread

	if(nGlobalIndex < nNumofTrainingSamples)
	{
		//update YiFValue
		pfYiFValue[nGlobalIndex] += (fsY1AlphaDiff * pfHessianRow1[nGlobalIndex] + fsY2AlphaDiff * pfHessianRow2[nGlobalIndex]);
	}

}
