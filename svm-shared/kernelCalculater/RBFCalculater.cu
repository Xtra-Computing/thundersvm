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
bool CRBFKernel::GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, float_point *pfHessianDiag)
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
bool CRBFKernel::ComputeHessianRows(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevSelfDot,
									float_point *pfDevHessianRows, const int &nNumofCols, const int &nNumofDim,
									const int &nNumofRows, int nStartRow, int nStartCol)
{
	bool bReturn = true;

	ComputeHessianMatrix(pfDevSamples, pfDevTransSamples, pfDevSelfDot, pfDevHessianRows,
						 nNumofCols, nNumofDim, nNumofRows, nStartRow, nStartCol);

//	float *pMatrix1 = new float[nNumofCols * nNumofRows];

//	checkCudaErrors(cudaMemcpy(pMatrix1, pfDevHessianRows, sizeof(float) * nNumofCols * nNumofRows, cudaMemcpyDeviceToHost));

/*	int nBlockSize = 0;
	dim3 dimGrid;
	GetGPUSpec(dimGrid, nBlockSize, nNumofCols, nNumofRows);
	assert(nBlockSize >= 0);
	cout << "gamma=" << m_fGamma << endl;
	cout << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << endl;
	cout << nBlockSize << " " << nNumofCols << " " << nNumofRows << endl;
	RBFKernel<<<dimGrid, nBlockSize, nBlockSize * sizeof(float_point)>>>(pfDevSamples,
			pfDevTransSamples, pfDevHessianRows, nNumofCols, nNumofDim, nNumofRows, 0, m_fGamma);
	cudaDeviceSynchronize();
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "cuda error in ComputeHessianRows" << endl;
		exit(0);
	}

	float *pMatrix2 = new float[nNumofCols * nNumofRows];
	checkCudaErrors(cudaMemcpy(pMatrix2, pfDevHessianRows, sizeof(float) * nNumofCols * nNumofRows, cudaMemcpyDeviceToHost));
	for(int i = 0; i < nNumofCols * nNumofRows; i++)
	{
		if(abs(pMatrix2[i] - pMatrix1[i]) > 0.001)
		{
			cout << "diff: " << i << "; " << pMatrix1[i] << " v.s. " << pMatrix2[i] << endl;
			exit(0);
		}
	}
*/
	return bReturn;
}

/*
 * @brief: compute a certain # of rows of the Hessian Matrix by RBF function
 * @param: pfDevSamples: a device pointer to the whole samples. These samples indicate which rows are computed in this round
 * @param: pfDevTransSamples: a device pointer to the whole samples with transposition
 * @param: pfdevHessianRows: a device pointer to a certain # of Hessian Matrix rows to be computed
 * @param: nNumofRows:
 */
bool CRBFKernel::ComputeHessianMatrix(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevSelfDot,
									  float_point *pfDevHessianRows, const int &nNumofCols, const int &nNumofDim,
									  const int &nNumofRows, int nStartRow, int nStartCol)
{
	bool bReturn = true;

//	cublasSgemm ('n', 'n', nNumofRows, nNumofRows, nNumofDim, 1,
//			pfDevTransSamples, nNumofRows, pfDevSamples, nNumofDim,
//				 0, pfDevHessianRows, nNumofRows);
	cublasSgemm ('n', 'n', nNumofCols, nNumofRows, nNumofDim, 1,
				 pfDevTransSamples, nNumofCols, pfDevSamples, nNumofDim,
				 0, pfDevHessianRows, nNumofCols);
/*	 float *pHostRow100 = new float[nNumofCols];
	 cudaMemcpy(pHostRow100, pfDevHessianRows + 299 * nNumofCols, sizeof(float) * nNumofCols, cudaMemcpyDeviceToHost);*/

/*	 float *pDevRow100;
	 cudaMalloc((void**)&pDevRow100, sizeof(float) * nNumofCols);
	cublasSgemm ('n', 'n', nNumofCols, 1, nNumofDim, 1,
				 pfDevTransSamples, nNumofCols, pfDevSamples + 299 * nNumofDim, nNumofDim,
				 0, pDevRow100, nNumofCols);
		 float *pHostRow100 = new float[nNumofCols];
		 cudaMemcpy(pHostRow100, pDevRow100, sizeof(float) * nNumofCols, cudaMemcpyDeviceToHost);
		 for(int i = 0; i < 900; i++)
		 {
			 printf("%.9f ", pHostRow100[i]);
			 if(i != 0 && i % 10 == 0)
				 cout << endl;
		 }
		 cout << endl;

		 float *pRow100;
		 cudaMalloc((void**)&pRow100, sizeof(float) * nNumofCols);
		 cublasSgemv ('n', nNumofCols, nNumofDim, 1,
				 	  pfDevTransSamples, nNumofCols, pfDevSamples + 299 * nNumofDim, 1,
				 	  0, pRow100, 1);
		 float *pHost2Row100 = new float[nNumofCols];
		 	 cudaMemcpy(pHost2Row100, pRow100, sizeof(float) * nNumofCols, cudaMemcpyDeviceToHost);
		 	 for(int i = 0; i < 900; i++)
		 	 {
		 		 if(pHost2Row100[i] != pHostRow100[i])
		 		 {
		 			 printf("%.9f or %.9f\n", pHost2Row100[i] , pHostRow100[i]);
		 			 if(i > 10)
		 				 exit(0);
		 		 }

		 	 }

		 	 for(int i = 0; i < 900; i++)
		 	 {
		 		 printf("%.9f ", pHost2Row100[i]);
		 		 if(i != 0 && i % 10 == 0)
		 			 cout << endl;
		 	 }
*/
/*	float *pSample299 = new float[nNumofDim];
	cudaMemcpy(pSample299, pfDevSamples + 299* nNumofDim, sizeof(float) * nNumofDim, cudaMemcpyDeviceToHost);
	 for(int i = 0; i < 100; i++)
	 {
		 printf("%.9f ", pSample299[i]);
		 if(i != 0 && i % 10 == 0)
			 cout << endl;
	 }
	 cout << endl;
*/
/*	float *pRow100;
	cudaMalloc((void**)&pRow100, sizeof(float) * nNumofCols);
	 cublasSgemv ('n', nNumofCols, nNumofDim, 1,pfDevTransSamples, nNumofCols,
			 	  pfDevSamples + 299 * nNumofDim, 1, 0, pRow100, 1);
	 float *pHostRow100 = new float[nNumofCols];
	 cudaMemcpy(pHostRow100, pRow100, sizeof(float) * nNumofCols, cudaMemcpyDeviceToHost);
	 for(int i = 0; i < 900; i++)
	 {
		 printf("%.9f ", pHostRow100[i]);
		 if(i != 0 && i % 10 == 0)
			 cout << endl;
	 }
	 cout << endl;
*/
/*   float v100;
    cudaMemcpy(&v100, pfDevHessianRows+299*nNumofCols + 100, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "v100=" << v100 << "; index=" << 299*nNumofCols + 100 << endl;

*/

//	cublasSgemm ('n', 'n', nNumofRows, nNumofCols, nNumofDim, 1,
//				 pfDevSamples, nNumofRows, pfDevTransSamples, nNumofDim,
//				 0, pfDevHessianRows, nNumofRows);//at once

//	cublasSgemm ('n', 'n', nNumofRows, nNumofCols, nNumofDim, 1,
//				 pfDevTransSamples, nNumofDim, pfDevSamples, nNumofRows,
//				 0, pfDevHessianRows, nNumofRows);

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

	//cout << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << " " << nBlockSize << endl;
	//exit(0);
	assert(nBlockSize >= 0);
//	cout << "gamma=" << m_fGamma << endl;
///################# problem here. nNumofCols should be nNumofTotalSample
	ObtainRBFKernel<<<dimGrid, nBlockSize>>>(pfDevHessianRows, pfDevSelfDot, nNumofCols, nNumofRows, m_fGamma, nStartRow, nStartCol);
	cudaDeviceSynchronize();
//	UpdateDiag<<<dimGrid, nBlockSize>>>(pfDevHessianRows, nNumofSamples, nNumofRows);

	/*float *pMatrix = new float[nNumofSamples * nNumofRows];
	cudaMemcpy(pMatrix, pfDevHessianRows, sizeof(float) * nNumofSamples * nNumofRows, cudaMemcpyDeviceToHost);
	for(int i = 0; i < nNumofSamples; i++)
	{
		for(int j = 0; j < nNumofRows; j++)
		{
			if(abs(pMatrix[i * nNumofRows + j] - pMatrix[j * nNumofSamples + i]) > 0.001)
			{
				cout << pMatrix[i * nNumofRows + j] << " != " << pMatrix[j * nNumofSamples + i] << endl;
				exit(0);
			}
		}
	}*/

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
bool CRBFKernel::ComputeHessianRowsByCPU(float_point *pfSamples, float_point *pfHessianRows,
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
				float_point fDiff = pfSamples[i * nNumofDim + j] - pfSamples[k * nNumofDim + j];
				pfHessianRows[k * nNumofSamples + i] += fDiff * fDiff;
			}
			pfHessianRows[k * nNumofSamples + i] *= m_fGamma;
			pfHessianRows[k * nNumofSamples + i] = -pfHessianRows[k * nNumofSamples + i];
			pfHessianRows[k * nNumofSamples + i] = exp(pfHessianRows[k * nNumofSamples + i]);
		}
	}

	return bReturn;
}
