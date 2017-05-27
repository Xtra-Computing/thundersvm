#include "kernelCalculater.h"
#include "../my_assert.h"
#include "cublas.h"
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <sys/time.h>

/* *
 /*
 * @brief: get Hessian diagonal
 */
bool CRBFKernel::GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, real *pfHessianDiag)
{
	bool bReturn = false;

	for(int i = 0; i < nNumofTrainingSamples; i++)
	{
		//the value of diagonal of RBF kernel is always 1
		pfHessianDiag[i] = 1.0;
	}

	return bReturn;
}

/*
 * @brief: compute a certain # of rows of the Hessian Matrix by RBF function
 * @param: pfDevSamples: a device pointer to the whole samples. These samples indicate which rows are computed in this round
 * @param: pfDevTransSamples: a device pointer to the whole samples with transposition
 * @param: pfdevHessianRows: a device pointer to a certain # of Hessian Matrix rows to be computed
 * @param: nNumofRows:
 */
bool CRBFKernel::ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevSelfDot,
									real *pfDevHessianRows, const int &nNumofCols, const int &nNumofDim,
									const int &nNumofRows, int nStartRow, int nStartCol)
{
	bool bReturn = true;

	ComputeHessianMatrix(pfDevSamples, pfDevTransSamples, pfDevSelfDot, pfDevHessianRows,
						 nNumofCols, nNumofDim, nNumofRows, nStartRow, nStartCol);

	return bReturn;
}

/*
 * @brief: compute a certain # of rows of the Hessian Matrix by RBF function
 * @param: pfDevSamples: a device pointer to the whole samples. These samples indicate which rows are computed in this round
 * @param: pfDevTransSamples: a device pointer to the whole samples with transposition
 * @param: pfdevHessianRows: a device pointer to a certain # of Hessian Matrix rows to be computed
 * @param: nNumofRows:
 */
bool CRBFKernel::ComputeHessianMatrix(real *pfDevSamples, real *pfDevTransSamples, real *pfDevSelfDot,
									  real *pfDevHessianRows, const int &nNumofCols, const int &nNumofDim,
									  const int &nNumofRows, int nStartRow, int nStartCol)
{
	bool bReturn = true;

//	cublasSgemm ('n', 'n', nNumofRows, nNumofRows, nNumofDim, 1,
//			pfDevTransSamples, nNumofRows, pfDevSamples, nNumofDim,
//				 0, pfDevHessianRows, nNumofRows);

	/* Compute dot product of every two instances.
	 * COLUMN MAJOR format is used.
	 * The following function equals to: pfDevHessianRows = 1 * pfDevTransSamples * pfDevSamples + 0 * pfDevHessianRows
	 */
	cublasSgemm ('n', 'n', nNumofCols, nNumofRows, nNumofDim, 1,
				 pfDevTransSamples, nNumofCols, pfDevSamples, nNumofDim,
				 0, pfDevHessianRows, nNumofCols);

	//compute the kernel values
	int nBlockSize = 0;
	int nTotal = nNumofCols * nNumofRows;
	nBlockSize = ((nTotal > BLOCK_SIZE) ? BLOCK_SIZE : nTotal);
	int nNumofBlocks = Ceil(nTotal, nBlockSize);

	int nGridDimY = 0;
	nGridDimY = Ceil(nNumofBlocks, NUM_OF_BLOCK);
	int nGridDimX = 0;
	if(nNumofBlocks > NUM_OF_BLOCK)
		nGridDimX = NUM_OF_BLOCK;
	else
		nGridDimX = nNumofBlocks;
	dim3 dimGrid(nGridDimX, nGridDimY);

	assert(nBlockSize >= 0);
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "cuda error calling ObtainRBFKernel" << endl;
		exit(0);
	}
//	cout << "gamma=" << m_fGamma << endl;
///################# problem here. nNumofCols should be nNumofTotalSample
	ObtainRBFKernel<<<dimGrid, nBlockSize>>>(pfDevHessianRows, pfDevSelfDot, nNumofCols, nNumofRows, m_fGamma, nStartRow, nStartCol);
	cudaDeviceSynchronize();
//	UpdateDiag<<<dimGrid, nBlockSize>>>(pfDevHessianRows, nNumofSamples, nNumofRows);

	cudaDeviceSynchronize();
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "cuda error in ComputeHessianRows" << endl;
		exit(0);
	}

	return bReturn;
}

/*
 * @brief: compute Hessian rows by CPU. For testing GPU ComputeHessianRows purpose.
 */
bool CRBFKernel::ComputeHessianRowsByCPU(real *pfSamples, real *pfHessianRows,
								 	 	 const int &nNumofSamples, const int &nNumofDim,
								 	 	 const int &nStartRow)
{
	bool bReturn = false;

	for(int k = 0; k < nNumofSamples; k++)
	{
		for(int i = 0; i < nNumofSamples; i++)
		{
			for(int j = 0; j < nNumofDim; j++)
			{
				real fDiff = pfSamples[i * nNumofDim + j] - pfSamples[k * nNumofDim + j];
				pfHessianRows[k * nNumofSamples + i] += fDiff * fDiff;
			}
			pfHessianRows[k * nNumofSamples + i] *= m_fGamma;
			pfHessianRows[k * nNumofSamples + i] = -pfHessianRows[k * nNumofSamples + i];
			pfHessianRows[k * nNumofSamples + i] = exp(pfHessianRows[k * nNumofSamples + i]);
		}
	}

	return bReturn;
}
