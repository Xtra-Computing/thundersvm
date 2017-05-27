/*
 * SigmoidCalculater.cu
 *
 * Created on: 28/05/2013
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/


#include "kernelCalculater.h"
#include "kernelCalGPUHelper.h"
#include "../my_assert.h"

/*
 * @brief: compute a certain # of rows of the Hessian Matrix by Polynomial function
 * @param: pfDevSamples: a device pointer to the whole samples. These samples indicate which rows are computed in this round
 * @param: pfDevTransSamples: a device pointer to the whole samples with transposition
 * @param: pfdevHessianRows: a device pointer to a certain # of Hessian Matrix rows to be computed
 * @param: nNumofSamples: indicates the length of pfDevTransSamples
 * @param: nNumofRows: indicates the length of pfDevSamples
 */
bool CSigmoidKernel::ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
									const int &nNumofSamples, const int &nNumofDim,
									const int &nNumofRows, const int &nStartRow)
{
	bool bReturn = true;

	int nBlockSize = 0;
	dim3 dimGrid;
	GetGPUSpec(dimGrid, nBlockSize, nNumofSamples, nNumofRows);
	assert(nBlockSize >= 0);
	SigmoidKernel<<<dimGrid, nBlockSize, nBlockSize * sizeof(real)>>>(pfDevSamples,
			pfDevTransSamples, pfDevHessianRows, nNumofSamples, nNumofDim, nStartRow, m_fCoef);

	cudaDeviceSynchronize();
	assert(cudaGetLastError() == cudaSuccess);

	return bReturn;
}



