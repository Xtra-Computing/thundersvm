/**
 * LinearCalGPUHelper.cu
 * Created on: May 14, 2013
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#include "kernelCalGPUHelper.h"
/*
 * @brief: compute one Hessian row
 * @param: pfDevSamples: data of samples. One dimension array represents a matrix
 * @param: pfDevTransSamples: transpose data of samples
 * @param: pfDevHessianRows: a Hessian row. the final result of this function
 * @param: nNumofSamples: the number of samples
 * @param: nNumofDim: the number of dimensions for samples
 * @param: nStartRow: the Hessian row to be computed
 */
__device__ void LinearOneRow(real *pfDevSamples, real *pfDevTransSamples,
						  real *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow)
{
	int nThreadId = threadIdx.x;
	int nBlockSize = blockDim.x;
	int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
	extern __shared__ real fSampleValue[];

	int nTempPos = 0;
	real fKernelValue = 0;
	int nRemainDim = nNumofDim;

	if(nThreadId >= nBlockSize)
	{
		return;
	}

	//when the # of dimension is huge, we process a part of dimension. one thread per kernel value
	for(int j = 0; nRemainDim > 0; j++)
	{
		//starting position of a sample, + j * nNumofValuesInShdMem is for case that dimension is too large
		nTempPos = nStartRow * nNumofDim + j * nBlockSize;

		//in the case, nThreadId < nMaxNumofThreads
		//load (part of or all of) the sample values into shared memory
		if(nThreadId < nRemainDim)
		{
			fSampleValue[nThreadId] = pfDevSamples[nTempPos + nThreadId];
		}
		__syncthreads(); //synchronize threads within a block

		/* start compute kernel value */
		if(nGlobalIndex < nNumofSamples)
		{
			real fTempSampleValue;
			//when the block size is larger than remaining dim, k is bounded by nRemainDim
			//when the nRemainDim is larger than block size, k is bounded by nBlockSize
			for(int k = 0; (k < nBlockSize) && (k < nRemainDim); k++)
			{
				nTempPos = (nNumofDim - nRemainDim + k) * nNumofSamples + nGlobalIndex;
				fTempSampleValue = pfDevTransSamples[nTempPos]; //transpose sample
				fKernelValue += (fSampleValue[k] * fTempSampleValue);
			}
		}

		nRemainDim -= nBlockSize;
		//synchronize threads within block to avoid modifying shared memory
		__syncthreads();
	}//end computing one kernel value

	//load the element to global
	if(nGlobalIndex < nNumofSamples)
	{
		pfDevHessianRows[nGlobalIndex] = fKernelValue;
	}
}


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void LinearKernel(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nStartRow)
{
	real *pfDevTempHessianRow;
	//pointer to a hessian row
	pfDevTempHessianRow = pfDevHessianRows + blockIdx.z * nNumofSamples;
	LinearOneRow(pfDevSamples, pfDevTransSamples, pfDevTempHessianRow,
				 nNumofSamples, nNumofDim, nStartRow + blockIdx.z);

}
