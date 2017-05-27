/**
 * hessianIO.cu
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#include "deviceHessian.h"
#include <helper_cuda.h>
#include <sys/time.h>
#include "../gpu_global_utility.h"
#include "../constant.h"
#include "cublas.h"
#include "../storageManager.h"

using std::endl;

long lIO_timer = 0;
long lIO_counter = 0;

CKernelCalculater *DeviceHessian::m_pKernelCalculater = NULL;

/*
 * @brief: read Hessian diagonal from file (for RBF kernel, we assign 1.0 directly)
 */
bool DeviceHessian::GetHessianDiag(const string &strFileName, const int &nNumofInstance, real *pfHessianDiag)
{
	bool bReturn = true;

	assert(nNumofInstance > 0);
	if(m_pKernelCalculater->GetType() == RBFKERNEL)
		m_pKernelCalculater->GetHessianDiag(strFileName, nNumofInstance, pfHessianDiag);
	else
	{
		if(m_nRowStartPos1 != -1)
		{
			assert(m_nRowStartPos1 >= 0 && m_nRowEndPos1 > 0);
			for(int i = m_nRowStartPos1; i <= m_nRowEndPos1; i++)
			{
				assert(i < m_nTotalNumofInstance);
				pfHessianDiag[i - m_nRowStartPos1] = m_pfHessianDiag[i];
			}
		}

		if(m_nRowStartPos2 != -1)
		{
			int nOffset = 0;
			if(m_nRowEndPos1 != -1)
			{
				nOffset = (m_nRowEndPos1 + 1);
			}

			assert(m_nRowStartPos2 >= 0 && m_nRowEndPos2 > 0);
			for(int i = m_nRowStartPos2; i <= m_nRowEndPos2; i++)
			{
				assert(i - m_nRowStartPos2 + nOffset < nNumofInstance && (i - m_nRowStartPos2 + nOffset) >= 0);
				pfHessianDiag[i - m_nRowStartPos2 + nOffset] = m_pfHessianDiag[i];
			}
		}
	}

	return bReturn;
}

/*
 * @brief: get the size for each batch write
 * @return: the number of Hessian rows for each write
 */
int DeviceHessian::GetNumofBatchWriteRows()
{
	int nReturn = 0;

	if(m_nTotalNumofInstance > 0)
	{
		//initialize cache
		StorageManager *manager = StorageManager::getManager();
		long long nMaxNumofFloatPoint = manager->GetFreeGPUMem();

		nReturn = (nMaxNumofFloatPoint / (m_nTotalNumofInstance * sizeof(real)));
	}
	if(nReturn > m_nTotalNumofInstance)
	{
		nReturn = m_nTotalNumofInstance;
	}

	return nReturn;
}

/*
 * @brief: compute hessian sub matrix
 */
bool DeviceHessian::ComputeSubHessianMatrix(real *pfDevTotalSamples, real *pfDevTransSamples,real *pfDevSelfDot,
											real *pfDevNumofHessianRows, int nSubMatrixCol, int nSubMatrixRow,
											int nStartRow, int nStartCol)
{
	bool bReturn = true;
	if(m_nNumofDim > NORMAL_NUMOF_DIM)
	{
		cerr << "the number of dimension is very large" << endl;
	}

	//compute a few rows of Hessian matrix
	long long nHessianRowsSpace = nSubMatrixRow * (long long)nSubMatrixCol;
	long long nHessianRowSpaceInByte = sizeof(real) * nHessianRowsSpace;
	checkCudaErrors(cudaMemset(pfDevNumofHessianRows, 0, nHessianRowSpaceInByte));

	timeval t1, t2;
	real elapsedTime;
	gettimeofday(&t1, NULL);
//	cout << "computing " << nSubMatrixRow << " sub Hessian rows which have " << nSubMatrixCol << " column each" << endl;
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "cuda error before ComputeHessianRows" << endl;
		exit(0);
	}

	//pfDevTotalSamples is for row width (|); pfDevTransSamples is for col width (-)
	bool bComputeRows = m_pKernelCalculater->ComputeHessianRows(pfDevTotalSamples, pfDevTransSamples, pfDevSelfDot, pfDevNumofHessianRows,
																nSubMatrixCol, m_nNumofDim, nSubMatrixRow, nStartRow, nStartCol);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
//	cout << "computing kernel time " << elapsedTime << " ms.\n";
	assert(bComputeRows == true);

	return bReturn;
}

/**
 * @brief: compute the whole hessian matrix at once
 */
void DeviceHessian::ComputeHessianAtOnce(real *pfTotalSamples, real *pfTransSamples, real *pfSelfDot)
{
	//compute a few rows of Hessian matrix
	real *pfDevTransSamples;
	real *pfDevNumofHessianRows;
	real *pfDevTotalSamples;
	real *pfDevSelfDot;
	long lSpace = (long)m_nNumofDim * m_nTotalNumofInstance;
	long lResult = (long)m_nTotalNumofInstance * m_nTotalNumofInstance;
	checkCudaErrors(cudaMalloc((void**)&pfDevTransSamples, sizeof(real) * lSpace));
	checkCudaErrors(cudaMalloc((void**)&pfDevTotalSamples, sizeof(real) * lSpace));
	checkCudaErrors(cudaMalloc((void**)&pfDevNumofHessianRows, sizeof(real) * lResult));
	checkCudaErrors(cudaMalloc((void**)&pfDevSelfDot, sizeof(real) * m_nTotalNumofInstance));
	checkCudaErrors(cudaMemcpy(pfDevSelfDot, pfSelfDot, sizeof(real) * m_nTotalNumofInstance, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pfDevTotalSamples, pfTotalSamples,
					sizeof(real) * lSpace, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pfDevTransSamples, pfTransSamples,
					sizeof(real) * lSpace, cudaMemcpyHostToDevice));

	ComputeSubHessianMatrix(pfDevTotalSamples, pfDevTransSamples, pfDevSelfDot,
							pfDevNumofHessianRows, m_nTotalNumofInstance, m_nTotalNumofInstance, 0, 0);

	checkCudaErrors(cudaMemcpy(m_pfHessianRowsInHostMem, pfDevNumofHessianRows,
							  sizeof(real) * lResult, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(pfDevTotalSamples));
	checkCudaErrors(cudaFree(pfDevTransSamples));
	checkCudaErrors(cudaFree(pfDevNumofHessianRows));
	checkCudaErrors(cudaFree(pfDevSelfDot));

}

/*
 * @brief: compute Hessian matrix, write Hessian matrix and Hessian diagonal to two files respectively
 * @param: strHessianMatrixFileName: file name of a file storing hessian matrix (which serves as an output of this function)
 * @param: strDiagHessianFileName:	 file name of a file storing diagonal of hessian matrix
 * @param: v_v_DocVector: document vectors of all the training samples
 */

bool DeviceHessian::PrecomputeHessian(const string &strHessianMatrixFileName,
								 const string &strDiagHessianFileName,
								 vector<vector<real> > &v_v_DocVector)
{
	bool bReturn = true;

	m_nTotalNumofInstance = v_v_DocVector.size();
	m_nNumofDim = (v_v_DocVector.front()).size();
	m_nNumofHessianRowsToWrite = GetNumofBatchWriteRows();
	assert(m_nNumofHessianRowsToWrite != 0);

	//linear array for training samples
	long long nSpaceForSamples = (long long)m_nTotalNumofInstance * m_nNumofDim;
	real *pfTotalSamples = new real[nSpaceForSamples];
	memset(pfTotalSamples, 0, sizeof(real) * nSpaceForSamples);
	//copy samples to a linear array
	for(int i = 0; i < m_nTotalNumofInstance; i++)
	{
		//assign document vector to svm node
		for(int j = 0; j < m_nNumofDim; j++)
		{
			long long nIndex = (long long)i * m_nNumofDim + j;
			pfTotalSamples[nIndex] = v_v_DocVector[i][j];
			//pfTransSamples[j * m_nTotalNumofInstance + i] = v_v_DocVector[i][j];
		}
	}

//	v_v_DocVector.clear();
	real *pfTransSamples = new real[nSpaceForSamples];
	memset(pfTransSamples, 0, sizeof(real) * nSpaceForSamples);
	//copy samples to a linear array
	for(int i = 0; i < m_nTotalNumofInstance; i++)
	{
		//assign document vector to svm node
		for(int j = 0; j < m_nNumofDim; j++)
		{
			long long nIndex = (long long)j * m_nTotalNumofInstance + i;
			long long nIndex2 = (long long)i * m_nNumofDim + j;
			pfTransSamples[nIndex] = pfTotalSamples[nIndex2];
		}
	}

	//self dot product
	real *pfSelfDot = new real[m_nTotalNumofInstance];
	//copy samples to a linear array
	for(int i = 0; i < m_nTotalNumofInstance; i++)
	{
		//assign document vector to svm node
		real fTemp = 0;;
		for(int j = 0; j < m_nNumofDim; j++)
		{
			long long nIndex = (long long)i * m_nNumofDim + j;
			//fTemp += (v_v_DocVector[i][j] * v_v_DocVector[i][j]);
			fTemp += (pfTotalSamples[nIndex] * pfTotalSamples[nIndex]);
		}
		pfSelfDot[i] = fTemp;
	}

	//compute the minimum number of sub matrices that are required to calculate the whole Hessian matrix
	StorageManager *manager = StorageManager::getManager();
	int nNumofPartForARow = manager->PartOfRow(m_nTotalNumofInstance,m_nNumofDim);
	int nNumofPartForACol = manager->PartOfCol(nNumofPartForARow, m_nTotalNumofInstance, m_nNumofDim);

	cout << nNumofPartForARow << " parts of row; " << nNumofPartForACol << " parts of col.";
	cout.flush();

	//If the kernel matrix has been computed (for debugging your code), you can use 1/5 to 5/5 to save some computation
	//1/5
	pHessianFile = fopen(strHessianMatrixFileName.c_str(), "wb");
	if(pHessianFile == NULL)
	{
		cout << "open " << strHessianMatrixFileName << " failed" << endl;
		exit(0);
	}
	/**/
	/*********** process the whole matrix at once *****************/
	if(nNumofPartForARow == nNumofPartForACol && nNumofPartForACol == 1)
	{
		ComputeHessianAtOnce(pfTotalSamples, pfTransSamples, pfSelfDot);

		delete[] pfTotalSamples;
		delete[] pfTransSamples;
		delete[] pfSelfDot;
        fclose(pHessianFile);
		return true;
	}

	//open file to write. When the file is open, the content is empty


	//length for sub row
	int *pLenofEachSubRow = new int[nNumofPartForARow];
	int nAveLenofSubRow = Ceil(m_nTotalNumofInstance, nNumofPartForARow);
	for(int i = 0; i < nNumofPartForARow; i++)
	{
		if(i + 1 != nNumofPartForARow)
			pLenofEachSubRow[i] = nAveLenofSubRow;
		else
			pLenofEachSubRow[i] = m_nTotalNumofInstance - nAveLenofSubRow * i;
	}
	//length for sub row
	int *pLenofEachSubCol = new int[nNumofPartForACol];
	int nAveLenofSubCol = Ceil(m_nTotalNumofInstance, nNumofPartForACol);
	for(int i = 0; i < nNumofPartForACol; i++)
	{
		if(i + 1 != nNumofPartForACol)
			pLenofEachSubCol[i] = nAveLenofSubCol;
		else
			pLenofEachSubCol[i] = m_nTotalNumofInstance - nAveLenofSubCol * i;
	}

	/*********************start to compute the sub matrices******************/
	//variables on host side
	long long lMaxSubMatrixSize = (long long)nAveLenofSubCol * nAveLenofSubRow;
	long long nMaxTransSamplesInCol = (long long)m_nNumofDim * nAveLenofSubRow;
	long long nMaxSamplesInRow = (long long)m_nNumofDim * nAveLenofSubCol;
	real *pfSubMatrix = new real[lMaxSubMatrixSize];
	//float_point *pfSubMatrixRowMajor = new float_point[lMaxSubMatrixSize];
	real *pfTransSamplesForAColInSubMatrix;
	pfTransSamplesForAColInSubMatrix = new real[nMaxTransSamplesInCol];
	real *pfSamplesForARowInSubMatrix;
//	pfSamplesForARowInSubMatrix = new float_point[nMaxSamplesInRow];

	//compute a few rows of Hessian matrix
	real *pfDevTransSamples;
	real *pfDevNumofHessianRows;
	real *pfDevTotalSamples;
	real *pfDevSelfDot;
	checkCudaErrors(cudaMalloc((void**)&pfDevTransSamples, sizeof(real) * nMaxTransSamplesInCol));
	checkCudaErrors(cudaMalloc((void**)&pfDevTotalSamples, sizeof(real) * nMaxSamplesInRow));
	checkCudaErrors(cudaMalloc((void**)&pfDevNumofHessianRows, sizeof(real) * lMaxSubMatrixSize));
	checkCudaErrors(cudaMalloc((void**)&pfDevSelfDot, sizeof(real) * m_nTotalNumofInstance));
	checkCudaErrors(cudaMemcpy(pfDevSelfDot, pfSelfDot, sizeof(real) * m_nTotalNumofInstance, cudaMemcpyHostToDevice));

	//compuate sub matrix
	for(int iPartofCol = 0; iPartofCol < nNumofPartForACol; iPartofCol++)
	{
		//get sub matrix row
		int nSubMatrixRow;
		if(iPartofCol == nNumofPartForACol - 1)
			nSubMatrixRow = pLenofEachSubCol[iPartofCol];
		else
			nSubMatrixRow = nAveLenofSubCol;

		for(int jPartofRow = 0; jPartofRow < nNumofPartForARow; jPartofRow++)
		{
			int nSubMatrixCol;
			//get sub matrix column
			if(jPartofRow == nNumofPartForARow - 1)
				nSubMatrixCol = pLenofEachSubRow[jPartofRow];
			else
				nSubMatrixCol = nAveLenofSubRow;

//			cout << "row= " << nSubMatrixRow << " col= " << nSubMatrixCol << endl;
//			cout << ".";
//			cout.flush();
			//allocate memory for this sub matrix
			long long nHessianSubMatrixSpace = nSubMatrixRow * nSubMatrixCol;
			memset(pfSubMatrix, 0, sizeof(real) * nHessianSubMatrixSpace);


			//get the copies of sample data for computing sub matrix

			//sample for sub matrix rows

			/*for(int d = 0; d < m_nNumofDimensions; d++)
			{
				memcpy(pfSamplesForARowInSubMatrix + d * nSubMatrixRow,
						pfTotalSamples + iPartofCol * nAveLenofSubCol + d * m_nTotalNumofSamples,
					   sizeof(float_point) * nSubMatrixRow);
			}
			pfTransSamplesForAColInSubMatrix = pfTransSamples + jPartofRow * nAveLenofSubRow * m_nNumofDimensions;*/
			pfSamplesForARowInSubMatrix = pfTotalSamples + (long long)iPartofCol * nAveLenofSubCol * m_nNumofDim;
			for(int d = 0; d < m_nNumofDim; d++)
			{
				//for(int k = 0; k < nSubMatrixCol; k++)
				//{
					//pfTransSamplesForAColInSubMatrix[k + d * nSubMatrixCol] =
					//pfTransSamples[k + jPartofRow * nAveLenofSubRow + d * m_nTotalNumofSamples];
				long long nSampleIndex = (long long)jPartofRow * nAveLenofSubRow + (long long)d * m_nTotalNumofInstance;
				long long nIndexForSub = (long long)d * nSubMatrixCol;
					memcpy(pfTransSamplesForAColInSubMatrix + nIndexForSub,
						   pfTransSamples + nSampleIndex, sizeof(real) * nSubMatrixCol);
				//}
			}

			long long nSpaceForSamplesInRow = (long long)m_nNumofDim * nSubMatrixRow;
			long long nSpaceForSamplesInCol = (long long)m_nNumofDim * nSubMatrixCol;
			checkCudaErrors(cudaMemcpy(pfDevTotalSamples, pfSamplesForARowInSubMatrix,
							sizeof(real) * nSpaceForSamplesInRow, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(pfDevTransSamples, pfTransSamplesForAColInSubMatrix,
							sizeof(real) * nSpaceForSamplesInCol, cudaMemcpyHostToDevice));
//
			//compute the value of the sub matrix
			int nStartRow = iPartofCol * nAveLenofSubCol;
			int nStartCol = jPartofRow * nAveLenofSubRow;
			ComputeSubHessianMatrix(pfDevTotalSamples, pfDevTransSamples, pfDevSelfDot,
									pfDevNumofHessianRows, nSubMatrixCol, nSubMatrixRow,
									nStartRow, nStartCol);

			int nHessianRowsSpace = nSubMatrixRow * nSubMatrixCol;
			checkCudaErrors(cudaMemcpy(pfSubMatrix, pfDevNumofHessianRows,
							sizeof(real) * nHessianRowsSpace, cudaMemcpyDeviceToHost));

			//store the sub matrix
			//hessian sub matrix info
			SubMatrix subMatrix;
			subMatrix.nColIndex = jPartofRow * nAveLenofSubRow;
			subMatrix.nColSize = nSubMatrixCol;
			subMatrix.nRowIndex = iPartofCol * nAveLenofSubCol;
			subMatrix.nRowSize = nSubMatrixRow;
			SaveRows(pfSubMatrix, subMatrix);

		}
		cout << ".";
		cout.flush();
	}
	delete[] pfSubMatrix;
	delete[] pfTransSamplesForAColInSubMatrix;

	delete[] pLenofEachSubRow;
	delete[] pLenofEachSubCol;
	delete[] pfTotalSamples;
	delete[] pfTransSamples;
	delete[] pfSelfDot;

	//release memory on GPU
	checkCudaErrors(cudaFree(pfDevTotalSamples));
	checkCudaErrors(cudaFree(pfDevTransSamples));
	checkCudaErrors(cudaFree(pfDevNumofHessianRows));
	checkCudaErrors(cudaFree(pfDevSelfDot));
//4/5
fclose(pHessianFile);
//5/5 is in smoSolver.h
	return bReturn;
}

