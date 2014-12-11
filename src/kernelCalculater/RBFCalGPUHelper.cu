
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
__device__ void RBFOneRow(float_point *pfDevSamples, float_point *pfDevTransSamples,
						  float_point *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, float_point fGamma)
{
	int nThreadId = threadIdx.x;
	int nBlockSize = blockDim.x;
	int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
	extern __shared__ float_point fSampleValue[];

	int nTempPos = 0;
	float_point fKernelValue = 0;
	int nRemainDim = nNumofDim;

	if(nThreadId >= nBlockSize)
	{
		return;
	}

	//when the # of dimension is huge, we process a part of dimension. one thread per kernel value
	for(int j = 0; nRemainDim > 0; j++)
	{
		//if(nThreadId == 0)
		//{
		//starting position of a sample, + j * nNumofValuesInShdMem is for case that dimension is too large
		nTempPos = nStartRow * nNumofDim + j * nBlockSize;
		//}
		//__syncthreads();

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
			float_point fTempSampleValue;
			float_point fDiff;
			//when the block size is larger than remaining dim, k is bounded by nRemainDim
			//when the nRemainDim is larger than block size, k is bounded by nBlockSize
			for(int k = 0; (k < nBlockSize) && (k < nRemainDim); k++)
			{
				nTempPos = (nNumofDim - nRemainDim + k) * nNumofSamples + nGlobalIndex;
				fTempSampleValue = pfDevTransSamples[nTempPos]; //transpose sample
				fDiff = fSampleValue[k] - fTempSampleValue;

				fKernelValue += (fDiff * fDiff);
			}
		}

		nRemainDim -= nBlockSize;
		//synchronize threads within block to avoid modifying shared memory
		__syncthreads();
	}//end computing one kernel value

	//load the element to global
	if(nGlobalIndex < nNumofSamples)
	{
		fKernelValue = fKernelValue * fGamma;
		fKernelValue = -fKernelValue; //Gaussian kernel use "-gamma"
		if(sizeof(float_point) == sizeof(double))
			fKernelValue = expf(fKernelValue);
		else
			fKernelValue = __expf(fKernelValue);

		pfDevHessianRows[nGlobalIndex] = fKernelValue;
	}
}


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void RBFKernel(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nNumofRows, int nStartRow, float_point fGamma)
{
	float_point *pfDevTempHessianRow;

	//for(int i = 0; i < nNumofRows; i++)
	{
		//pointer to a hessian row
		//pfDevTempHessianRow = pfDevHessianRows + i * nNumofSamples;
		pfDevTempHessianRow = pfDevHessianRows + blockIdx.z * nNumofSamples;

		RBFOneRow(pfDevSamples, pfDevTransSamples, pfDevTempHessianRow,
				  nNumofSamples, nNumofDim, nStartRow + blockIdx.z, fGamma);
		//nStartRow++;//increase to next row
	} //end computing n rows of Hessian Matrix

}

//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void ObtainRBFKernel(float_point *pfDevHessianRows, float_point *pfDevSelfDot, int nNumofSamples,
								int nNumofRows, float_point fGamma, int nStartRow, int nStartCol)
{
	int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
	if(nGlobalIndex < nNumofSamples * nNumofRows)
	{
		int nRow = (nGlobalIndex / nNumofSamples + nStartRow);
		int nCol = (nGlobalIndex % nNumofSamples + nStartCol);


		float fKernelValue = (pfDevSelfDot[nRow] + pfDevSelfDot[nCol] -pfDevHessianRows[nGlobalIndex] * 2.f ) * fGamma;
		fKernelValue = -fKernelValue; //Gaussian kernel use "-gamma"
		if(sizeof(float_point) == sizeof(double))
			fKernelValue = exp(fKernelValue);
		else
			fKernelValue = expf(fKernelValue);

		//if(nGlobalIndex == 299 * nNumofSamples + 100)
		//	printf("%f, %f, %f, %f, %d\n", pfDevSelfDot[nCol], pfDevSelfDot[nRow], pfDevHessianRows[nGlobalIndex], fKernelValue, nGlobalIndex);

		pfDevHessianRows[nGlobalIndex] = fKernelValue;
	}
}
