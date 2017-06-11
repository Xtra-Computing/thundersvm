
#include "classificationKernel.h"
#include <cuda.h>
/*
 * @brief: compute partial kernel sum for classification, as sometimes the # of SVs is very large
 * @param: pfSVYiAlphaHessian: an array storing yi*alpha*k(sv,x), where sv is a support vector. (in the format: T1 sv1 sv2...T2 sv1 sv2...)
 * @param: pfLocalPartialSum: a block level sum of yi*alpha*k(sv,x)
 * @param: nReduceStepSize: the step size for each pair of sum
 */
__global__ void ComputeKernelPartialSum(real* pfSVYiAlhpaHessian, int nNumofSVs,
										real* pfLocalPartialSum, int nReduceStepSize)
{
	extern __shared__ real fLocalValue[];
	int nThreadId = threadIdx.x;
	//each support vector is associated with a thread
	int nSVGlobalId = blockDim.x * blockIdx.x + nThreadId;

	//skip those threads having larger thread ID than the # of SVs
	if(nNumofSVs <= nSVGlobalId)
	{
		return;
	}

	//load yi*alpha*k(sv,x) to shared memory
	fLocalValue[nThreadId] = pfSVYiAlhpaHessian[(blockIdx.y * nNumofSVs) + nSVGlobalId];
	__syncthreads();

	//start sum reduction
	for(int nOffset = nReduceStepSize; nOffset >= 1; nOffset = nOffset >> 1)
	{
		if((nThreadId < nOffset) && (nSVGlobalId + nOffset < nNumofSVs))
		{
			int nIndexofReduceValue = nThreadId + nOffset;
			fLocalValue[nThreadId] = fLocalValue[nThreadId] + fLocalValue[nIndexofReduceValue];
		}
		__syncthreads();
	}

	//partial sum of a testing sample
	if(nThreadId == 0)
	{
		//blockIdx.y is for distinguishing testing samples, as sometimes there may be more than one testing sample
		pfLocalPartialSum[blockIdx.x + gridDim.x * blockIdx.y] = fLocalValue[0];
	}
}

/*
 * @brief: compute global sum for each testing sample, and the final result
 * @param: pfClassificationResult: the result of the sum (the output of this function)
 * @param: fBias: the bias term of SVM
 * @param: pfPartialSum: the partial sum of block level
 */
__global__ void ComputeKernelGlobalSum(real *pfClassificationResult,
									   real fBias, real *pfPartialSum,
									   int nReduceStepSize)
{
	extern __shared__ real fLocalValue[];

	int nThreadId = threadIdx.x;
	fLocalValue[nThreadId] = pfPartialSum[blockDim.x * blockIdx.y + nThreadId];
	__syncthreads();

	for(int nOffset = nReduceStepSize; nOffset >= 1; nOffset = nOffset >> 1)
	{
		if(nThreadId < nOffset)
		{
			int nIndexofReduceValue = nThreadId + nOffset;
			if(nIndexofReduceValue < blockDim.x)
			{
				fLocalValue[nThreadId] = fLocalValue[nThreadId] + fLocalValue[nIndexofReduceValue];
			}
		}
		__syncthreads();
	}

	real fGlobalSum = fLocalValue[0];
	if(nThreadId == 0)
	{
		fGlobalSum -= fBias;
		//blockIdx.y is for distinguishing testing samples
		pfClassificationResult[blockIdx.y] = fGlobalSum;
	}
}


/*
 * @brief: compute multiplication of two vectors
 * @output: result stores in pfVector1
 */
__global__ void VectorMul(real *pfVector1, int *pnVector2, int nNumofDim)
{
	int nThreadId = threadIdx.x;
	int nGlobalId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + nThreadId;
	if(nGlobalId >= nNumofDim)
	{
		return;
	}

	pfVector1[nGlobalId] = pfVector1[nGlobalId] * pnVector2[nGlobalId];
}

/*
 * @brief: compute a vector is multiplied by a matrix. This function does part of vector-matrix multiplication
 */
__global__ void VectorMatrixMul(real *pfVector, real *pfMatrix, int nNumofRow, int nNumofCol)
{
	int nThreadId = threadIdx.x;
	//a few blocks compute a row of the matrix
	int nSVId = blockIdx.x * blockDim.x + nThreadId;
	//row is determined be blockIdx.y
	int nElementPosInMatrix = blockIdx.y * nNumofCol + nSVId;
	if(nSVId >= nNumofCol)
	{
		return;
	}
	pfMatrix[nElementPosInMatrix] = pfMatrix[nElementPosInMatrix] * pfVector[nSVId];

}
