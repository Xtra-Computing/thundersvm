/**
 * hessianIO.cu
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#include "hessianIO.h"
#include "boost/interprocess/file_mapping.hpp"
#include <helper_cuda.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include "../gpu_global_utility.h"
#include "../constant.h"
#include "cublas.h"

using std::endl;
//initialize the static variables for Hessian Operator
int CHessianIOOps::m_nTotalNumofInstance = 0;
int CHessianIOOps::m_nNumofDimensions = 0;
float_point* CHessianIOOps::m_pfHessianRowsInHostMem = NULL;
float_point* CHessianIOOps::m_pfHessianDiag = NULL;
int CHessianIOOps::m_nNumofCachedHessianRow = 0;
CKernelCalculater *CHessianIOOps::m_pKernelCalculater = NULL;

int CHessianIOOps::m_nRowStartPos1;
int CHessianIOOps::m_nRowEndPos1;
CFileOps* CHessianIOOps::m_fileOps;

long lIO_timer = 0;
long lIO_counter = 0;
/*
 * @brief: set data involved in Hessian Read Operation
 * @param: nStart1: the index of the first part of a row
 * @param: nEnd1: the index of the end of the first part of a row
 * @param: nStart2: the index of the second part of a row
 * @param: nEnd2: the index of the end of the second part of a row
 */
bool CHessianIOOps::SetInvolveData(const int &nStart1, const int &nEnd1, const int &nStart2, const int &nEnd2)
{
	bool bReturn = false;

	if(nStart1 >= m_nTotalNumofInstance || nEnd1 >= m_nTotalNumofInstance ||
	   nStart2 >= m_nTotalNumofInstance || nEnd2 >= m_nTotalNumofInstance)
	{
		return bReturn;
	}
	m_nRowStartPos1 = nStart1;
	m_nRowEndPos1 = nEnd1;
	m_nRowStartPos2 = nStart2;
	m_nRowEndPos2 = nEnd2;

	return bReturn;
}
/*
 * @brief: get the size for each batch write
 * @return: the number of Hessian rows for each write
 */
int CHessianIOOps::GetNumofBatchWriteRows()
{
	int nReturn = 0;

	if(m_nTotalNumofInstance > 0)
	{
		//initialize cache
		size_t nFreeMemory,nTotalMemory;
		cuMemGetInfo(&nFreeMemory,&nTotalMemory);
		int nMaxNumofFloatPoint = nFreeMemory / sizeof(float_point);
		nMaxNumofFloatPoint *= 0.8;//use 80% of the free memory for hessian matrix calculation

		nReturn = (nMaxNumofFloatPoint / (m_nTotalNumofInstance * sizeof(float_point)));
	}
	if(nReturn > m_nTotalNumofInstance)
	{
		nReturn = m_nTotalNumofInstance;
	}

	return nReturn;
}

/*
 * @brief: allocate memory for reading content from file
 */
bool CHessianIOOps::AllocateBuffer(int nNumofRows)
{
	bool bReturn = false;

	if(nNumofRows < 1)
	{
		cerr << "error in hessian ops: allocate buffer failed!" << endl;
		return bReturn;
	}
	bReturn = true;
	m_pfHessianRows = new float_point[m_nTotalNumofInstance * nNumofRows];

	return bReturn;
}

/*
 * @brief: release buffer from reading hessian rows
 */
bool CHessianIOOps::ReleaseBuffer()
{
	if(m_pfHessianRows == NULL)
	{
		cerr << "buffer to be released is empty!" << endl;
		return false;
	}
	delete[] m_pfHessianRows;
	return true;
}

/*
 * @brief: read Hessian diagonal from file (for RBF kernel, we assign 1.0 directly)
 */
bool CHessianIOOps::GetHessianDiag(const string &strFileName, const int &nNumofInstance, float_point *pfHessianDiag)
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

void CHessianIOOps::ReadDiagFromHessianMatrix()
{
	float_point *hessianRow = new float_point[m_nTotalNumofInstance];
	if(m_pKernelCalculater->GetType() != RBFKERNEL)
	{
		FILE *readIn = fopen(HESSIAN_FILE, "rb");
		for(int i = 0; i < m_nTotalNumofInstance; i++)
		{
			//if the hessian row is in host memory
			if(m_nNumofCachedHessianRow > i)
			{
				long long nIndexofFirstElement = (long long) i * m_nTotalNumofInstance + i;
				m_pfHessianDiag[i] = m_pfHessianRowsInHostMem[nIndexofFirstElement];
			}
			else //the hessian row is in SSD
			{
				int nIndexInSSD = i - m_nNumofCachedHessianRow;
				ReadHessianFullRow(readIn, nIndexInSSD, 1, hessianRow);
				m_pfHessianDiag[i] = hessianRow[i];
			}
		}
		fclose(readIn);
	}
	delete[] hessianRow;
}

bool CHessianIOOps::MapIndexToHessian(int &nIndex)
{
	bool bReturn = false;
	//check input parameter
	int nTempIndex = nIndex;
	if(nIndex < 0 || (nIndex > m_nRowEndPos1 && nIndex > m_nRowEndPos2))
	{
		cerr << "error in MapIndexToHessian: invalid input parameter" << endl;
		exit(0);
	}

	bReturn = true;
	if(m_nRowStartPos1 != -1)
	{
		if(nIndex <= m_nRowEndPos1)
		{
			return bReturn;
		}
		else
		{
			nTempIndex = nIndex + (m_nRowStartPos2 - m_nRowEndPos1 - 1);
			if(nTempIndex < nIndex || nTempIndex > m_nRowEndPos2)
			{
				cerr << "error in MapIndexToHessian" << endl;
				exit(0);
			}
		}
	}
	else if(m_nRowStartPos2 != -1)
	{
		nTempIndex = nIndex + m_nRowStartPos2;
		if(nTempIndex > m_nRowEndPos2)
		{
			cerr << "error in MapIndexToHessian" << endl;
			exit(0);
		}
	}
	else
	{
		cerr << "error in MapIndexToHessian: m_nStart1 & 2 equal to -1" << endl;
		exit(0);
	}

	nIndex = nTempIndex;
	return bReturn;
}

bool CHessianIOOps::ReadHessianRow(boost::interprocess::mapped_region *pRegion, const int &nIndexofRow, const int &nNumofInvolveElements,
								   float_point *pfHessianRow, int nNumofElementEachRowInCache)
{
	timespec time1, time2;
	clock_gettime(CLOCK_REALTIME, &time1);

	int nSizeofFirstPart = 0;
	if(m_nRowStartPos1 != -1)
	{
		nSizeofFirstPart = m_nRowEndPos1 - m_nRowStartPos1 + 1;//the size of first part (include the last element of the part)
		long long nIndexofFirstElement = (long long)nIndexofRow * m_nTotalNumofInstance + m_nRowStartPos1;
		if(m_nNumofCachedHessianRow > nIndexofRow)
		{
			memcpy(pfHessianRow, m_pfHessianRowsInHostMem + nIndexofFirstElement, nSizeofFirstPart * sizeof(float_point));
		}
		else
		{
			m_fileOps->ReadPartOfRowFromFile(pRegion, pfHessianRow, m_nTotalNumofInstance, nSizeofFirstPart, nIndexofFirstElement);
		}
	}
	int nSizeofSecondPart = 0;
	if(m_nRowStartPos2 != -1)
	{
		nSizeofSecondPart = m_nRowEndPos2 - m_nRowStartPos2 + 1;
		long long nIndexofFirstElement = (long long)nIndexofRow * m_nTotalNumofInstance + m_nRowStartPos2;
		if(m_nNumofCachedHessianRow > nIndexofRow)
		{
			memcpy(pfHessianRow + nSizeofFirstPart, m_pfHessianRowsInHostMem + nIndexofFirstElement, nSizeofSecondPart * sizeof(float_point));
		}
		else
		{
			m_fileOps->ReadPartOfRowFromFile(pRegion, pfHessianRow + nSizeofFirstPart, m_nTotalNumofInstance, nSizeofSecondPart, nIndexofFirstElement);
		}
	}

	//check valid read
	if(nSizeofSecondPart + nSizeofFirstPart != nNumofInvolveElements)
	{
		cerr << "warning: reading hessian rows has potential error" << endl;
	}

	clock_gettime(CLOCK_REALTIME, &time2);

	long lTemp = ((time2.tv_sec - time1.tv_sec) * 1e9 + (time2.tv_nsec - time1.tv_nsec));
	if(lTemp > 0)
	{
		lIO_timer += lTemp;
	}

	return true;
}

/*
 * @brief: read one full Hessian row from file
 * @return: true if read the row successfully
 */
bool CHessianIOOps::ReadHessianFullRow(FILE *&readIn, const int &nIndexofRow, int nNumofRowsToRead, float_point *pfFullHessianRow)
{
	bool bReturn = false;
	assert(readIn != NULL && nIndexofRow >= 0 && nIndexofRow < m_nTotalNumofInstance);

	//read the whole Hessian row
	bReturn = m_fileOps->ReadRowsFromFile(readIn, pfFullHessianRow, m_nTotalNumofInstance, nNumofRowsToRead, nIndexofRow);
	assert(bReturn != false && pfFullHessianRow != NULL);

	return bReturn;
}

/*
 * @brief: read a continuous part of a Hessian row. Note that the last element (nEndPos) is included in the sub row
 * @output: pfHessianSubRow: part of a Hessian row
 */
bool CHessianIOOps::ReadHessianSubRow(FILE *&readIn, const int &nIndexofRow,
		   	   	   	   	   	   	   	  const int &nStartPos, const int &nEndPos,
		   	   	   	   	   	   	   	  float_point *pfHessianSubRow)
{
	bool bReturn = false;
	if(readIn == NULL || nIndexofRow < 0 || nIndexofRow > m_nTotalNumofInstance ||
		nStartPos < 0    || nEndPos < 0 	|| nStartPos > m_nTotalNumofInstance   ||
		nEndPos > m_nTotalNumofInstance)
	{
		cerr << "error in ReadHessianSubRow: invalid param" << endl;
		return bReturn;
	}

	int nNumofHessianElements = nEndPos - nStartPos + 1;//the number of elements to read

	//read the whole Hessian row
	float_point *pfTempFullHessianRow = new float_point[m_nTotalNumofInstance];
	bool bReadRow =	ReadHessianFullRow(readIn, nIndexofRow, 1, pfTempFullHessianRow); //1 means that read one Hessian row
	if(bReadRow == false)
	{
		cerr << "error in ReadHessianRow" << endl;
		delete[] pfTempFullHessianRow;
		return bReturn;
	}

	//get sub row from a full Hessian row
	memcpy(pfHessianSubRow, pfTempFullHessianRow + nStartPos, sizeof(float_point) * nNumofHessianElements);

	delete[] pfTempFullHessianRow;

	bReturn = true;
	return bReturn;
}

/*
 * @brief: read a few Hessian rows; This functionality is required for initialised cache, and etc. Read from nStartRow to nEndRow (include the last row)
 * @param: nNumofInvolveElements: the number of involved elements of a row
 * @param: pfHessianRow: the space to store the hessian row(s)
 * @param: nNumofElementEachRowInCache: number of element of each row in pfHessian.
 * 		   Because of the memory alignment issue, this param is usually bigger than nNumofInvolveElements
 */
bool CHessianIOOps::ReadHessianRows(FILE *&readIn, const int &nStartRow, const int &nEndRow,
									const int &nNumofInvolveElements, float_point * pfHessianRow, int nNumOfElementEachRowInCache)
{
	bool bReturn = false;
	//check input parameters
	if(readIn == NULL || nStartRow > m_nTotalNumofInstance || nEndRow > m_nTotalNumofInstance || nEndRow < nStartRow)
	{
		cerr << "error in ReadHessianRows: invalid input parameters" << endl;
		return bReturn;
	}

	//start reading Hessain sub rows
	int nSizeofFirstPart = 0;
	if(m_nRowStartPos1 != -1)
	{
		nSizeofFirstPart = m_nRowEndPos1 - m_nRowStartPos1 + 1;//the size of first part (include the last element of the part)
	}
	int nSizeofSecondPart = 0;
	if(m_nRowStartPos2 != -1)
	{
		nSizeofSecondPart = m_nRowEndPos2 - m_nRowStartPos2 + 1;
	}

	//check valid read
	if(nSizeofSecondPart + nSizeofFirstPart != nNumofInvolveElements)
	{
		cerr << "warning: reading hessian rows has potential error" << endl;
	}
	//read Hessian rows at one time
	int nNumofRows = nEndRow - nStartRow + 1;
	//float_point *pfTempHessianRows = new float_point[m_nTotalNumofSamples * nNumofRows];
	bool bReadRow =	ReadHessianFullRow(readIn, nStartRow, nNumofRows,  m_pfHessianRows);
	if(bReadRow == false)
	{
		cerr << "error in ReadHessianRow" << endl;
		return bReturn;
	}

	bReturn = true;
	//read a full Hessian row
	int nHessianEndPos;
	float_point *pfTempFullHessianRow;// = new float_point[m_nTotalNumofSamples];
	for(int i = nStartRow; i <= nEndRow; i++)
	{
		pfTempFullHessianRow = m_pfHessianRows + (i - nStartRow) * m_nTotalNumofInstance;

		//read the first continuous part
		if(m_nRowStartPos1 != -1)
		{
			//first part is added to the end of current Hessian space in main memory
			nHessianEndPos = (i - nStartRow) * nNumOfElementEachRowInCache;//use number of elements each row instead of number of involve elements due to memory alignment
			memcpy(pfHessianRow + nHessianEndPos, pfTempFullHessianRow + m_nRowStartPos1, sizeof(float_point) * nSizeofFirstPart);
		}
		//read the second continuous part
		if(m_nRowStartPos2 != -1)
		{
			nHessianEndPos = (i - nStartRow) * nNumOfElementEachRowInCache + nSizeofFirstPart;
			memcpy(pfHessianRow + nHessianEndPos, pfTempFullHessianRow + m_nRowStartPos2, sizeof(float_point) * nSizeofSecondPart);
		}
	}
	//delete[] pfTempHessianRows;
	return bReturn;
}

/*
 * @brief: compute hessian sub matrix
 */
bool CHessianIOOps::ComputeSubHessianMatrix(float_point *pfDevTotalSamples, float_point *pfDevTransSamples,float_point *pfDevSelfDot,
											float_point *pfDevNumofHessianRows, int nSubMatrixCol, int nSubMatrixRow,
											int nStartRow, int nStartCol)
{
	bool bReturn = true;
	if(m_nNumofDimensions > NORMAL_NUMOF_DIM)
	{
		cerr << "the number of dimension is very large" << endl;
	}

	//compute a few rows of Hessian matrix
	int nHessianRowsSpace = nSubMatrixRow * nSubMatrixCol;
	checkCudaErrors(cudaMemset(pfDevNumofHessianRows, 0, sizeof(float_point) * nHessianRowsSpace));

	timeval t1, t2;
	float_point elapsedTime;
	gettimeofday(&t1, NULL);
//	cout << "computing " << nSubMatrixRow << " sub Hessian rows which have " << nSubMatrixCol << " column each" << endl;

	//pfDevTotalSamples is for row width (|); pfDevTransSamples is for col width (-)
	bool bComputeRows = m_pKernelCalculater->ComputeHessianRows(pfDevTotalSamples, pfDevTransSamples, pfDevSelfDot, pfDevNumofHessianRows,
																nSubMatrixCol, m_nNumofDimensions, nSubMatrixRow, nStartRow, nStartCol);
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
void CHessianIOOps::ComputeHessianAtOnce(float_point *pfTotalSamples, float_point *pfTransSamples, float_point *pfSelfDot)
{
	//compute a few rows of Hessian matrix
	float_point *pfDevTransSamples;
	float_point *pfDevNumofHessianRows;
	float_point *pfDevTotalSamples;
	float_point *pfDevSelfDot;
	long lSpace = (long)m_nNumofDimensions * m_nTotalNumofInstance;
	long lResult = (long)m_nTotalNumofInstance * m_nTotalNumofInstance;
	checkCudaErrors(cudaMalloc((void**)&pfDevTransSamples, sizeof(float_point) * lSpace));
	checkCudaErrors(cudaMalloc((void**)&pfDevTotalSamples, sizeof(float_point) * lSpace));
	checkCudaErrors(cudaMalloc((void**)&pfDevNumofHessianRows, sizeof(float_point) * lResult));
	checkCudaErrors(cudaMalloc((void**)&pfDevSelfDot, sizeof(float_point) * m_nTotalNumofInstance));
	checkCudaErrors(cudaMemcpy(pfDevSelfDot, pfSelfDot, sizeof(float_point) * m_nTotalNumofInstance, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pfDevTotalSamples, pfTotalSamples,
					sizeof(float_point) * lSpace, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pfDevTransSamples, pfTransSamples,
					sizeof(float_point) * lSpace, cudaMemcpyHostToDevice));

	ComputeSubHessianMatrix(pfDevTotalSamples, pfDevTransSamples, pfDevSelfDot,
							pfDevNumofHessianRows, m_nTotalNumofInstance, m_nTotalNumofInstance, 0, 0);

	checkCudaErrors(cudaMemcpy(m_pfHessianRowsInHostMem, pfDevNumofHessianRows,
							  sizeof(float_point) * lResult, cudaMemcpyDeviceToHost));

}

/*
 * @brief: compute Hessian matrix, write Hessian matrix and Hessian diagonal to two files respectively
 * @param: strHessianMatrixFileName: file name of a file storing hessian matrix (which serves as an output of this function)
 * @param: strDiagHessianFileName:	 file name of a file storing diagonal of hessian matrix
 * @param: v_v_DocVector: document vectors of all the training samples
 */

bool CHessianIOOps::WriteHessian(const string &strHessianMatrixFileName,
								 const string &strDiagHessianFileName,
								 vector<vector<float_point> > &v_v_DocVector)
{
	bool bReturn = true;

	m_nTotalNumofInstance = v_v_DocVector.size();
	m_nNumofDimensions = (v_v_DocVector.front()).size();
	m_nNumofHessianRowsToWrite = GetNumofBatchWriteRows();
	assert(m_nNumofHessianRowsToWrite != 0);

	//linear array for training samples
	long int nSpaceForSamples = (long int)m_nTotalNumofInstance * m_nNumofDimensions;
	float_point *pfTotalSamples = new float_point[nSpaceForSamples];
	memset(pfTotalSamples, 0, sizeof(float_point) * nSpaceForSamples);
	//copy samples to a linear array
	for(int i = 0; i < m_nTotalNumofInstance; i++)
	{
		//assign document vector to svm node
		for(int j = 0; j < m_nNumofDimensions; j++)
		{
			long int nIndex = (long int)i * m_nNumofDimensions + j;
			pfTotalSamples[nIndex] = v_v_DocVector[i][j];
			//pfTransSamples[j * m_nTotalNumofInstance + i] = v_v_DocVector[i][j];
		}
	}

	v_v_DocVector.clear();
	float_point *pfTransSamples = new float_point[nSpaceForSamples];
	memset(pfTransSamples, 0, sizeof(float_point) * nSpaceForSamples);
	//copy samples to a linear array
	for(int i = 0; i < m_nTotalNumofInstance; i++)
	{
		//assign document vector to svm node
		for(int j = 0; j < m_nNumofDimensions; j++)
		{
			long int nIndex = (long int)j * m_nTotalNumofInstance + i;
			long int nIndex2 = (long int)i * m_nNumofDimensions + j;
			pfTransSamples[nIndex] = pfTotalSamples[nIndex2];
		}
	}

	//self dot product
	float_point *pfSelfDot = new float_point[m_nTotalNumofInstance];
	//copy samples to a linear array
	for(int i = 0; i < m_nTotalNumofInstance; i++)
	{
		//assign document vector to svm node
		float_point fTemp = 0;;
		for(int j = 0; j < m_nNumofDimensions; j++)
		{
			long int nIndex = (long int)i * m_nNumofDimensions + j;
			//fTemp += (v_v_DocVector[i][j] * v_v_DocVector[i][j]);
			fTemp += (pfTotalSamples[nIndex] * pfTotalSamples[nIndex]);
		}
		pfSelfDot[i] = fTemp;
	}

	//compute the minimum number of sub matrices that are required to calculate the whole Hessian matrix
	int nMaxNumofFloatPoint = GetFreeGPUMem();
	int nMaxNumofAllowedSampleInGPU = ((nMaxNumofFloatPoint / (2 * m_nNumofDimensions)));//2 is for two copies of samples, original and transposed samples
//nMaxNumofAllowedSampleInGPU *= 0.8;	//use 80% of the available memory on GPU
	//compute the number of partitions in column
	int nNumofPartForARow = Ceil(m_nTotalNumofInstance, nMaxNumofAllowedSampleInGPU);
	if(nNumofPartForARow > 1)//if all the samples cannot fit into the GPU memory
		nNumofPartForARow = Ceil(nNumofPartForARow, 2);	//divided by 2 because in the extreme case, the row and column share 0 samples.
	//compute the number of partitions in row
nNumofPartForARow = 3;
	//in the worst case, the number of column equals to the number of row
	int nNumofPartForACol = nNumofPartForARow;//divide the matrix into sub matrices. default is the same for row and col
	if(nNumofPartForARow == 1)//can compute a hessian row once
	{//check the maximum number of rows for each computation
		//how many we can compute based on GPU memory constrain
		long nNumofFloatARow = m_nTotalNumofInstance * sizeof(float_point) / nNumofPartForARow;//space for a row
		long lRemainingNumofFloat = nMaxNumofFloatPoint - ((m_nTotalNumofInstance / nNumofPartForARow) * m_nNumofDimensions);

		long nMaxNumofHessianSubRow = lRemainingNumofFloat / nNumofFloatARow;
		//nMaxNumofHessianSubRow *= 0.8; //use 80% of the available memory on GPU
		if(nMaxNumofHessianSubRow < 0)
		{
			nMaxNumofHessianSubRow = nNumofPartForACol;
		}

		//how many we can compute based on RAM constrain
		struct sysinfo info;
		sysinfo(&info);
		long nNumofFloatPoint = info.freeram / sizeof(float_point);
		long nNumofHessianSubRowCPU = nNumofFloatPoint / nNumofFloatARow;

		nMaxNumofHessianSubRow = (nNumofHessianSubRowCPU < nMaxNumofHessianSubRow ? nNumofHessianSubRowCPU : nMaxNumofHessianSubRow);
		nNumofPartForACol = Ceil(m_nTotalNumofInstance, nMaxNumofHessianSubRow);

	}
//nNumofPartForACol = 500;

//	cout << nNumofPartForARow << " parts of row; " << nNumofPartForACol << " parts of col." << endl;

	/*********** process the whole matrix at once *****************/
	if(nNumofPartForARow == 1 && nNumofPartForACol == 1)
	{
		ComputeHessianAtOnce(pfTotalSamples, pfTransSamples, pfSelfDot);

		delete[] pfTotalSamples;
		delete[] pfTransSamples;
		delete[] pfSelfDot;
		return true;
	}

	//open file to write. When the file is open, the content is empty

//1/4
	FILE *writeOut;
writeOut = fopen(strHessianMatrixFileName.c_str(), "wb");
assert(writeOut != NULL);
/**/
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
	long int lMaxSubMatrixSize = (long int)nAveLenofSubCol * nAveLenofSubRow;
	long int nMaxTransSamplesInCol = (long int)m_nNumofDimensions * nAveLenofSubRow;
	long int nMaxSamplesInRow = (long int)m_nNumofDimensions * nAveLenofSubCol;
	float_point *pfSubMatrix = new float_point[lMaxSubMatrixSize];
	//float_point *pfSubMatrixRowMajor = new float_point[lMaxSubMatrixSize];
	float_point *pfTransSamplesForAColInSubMatrix;
	pfTransSamplesForAColInSubMatrix = new float_point[nMaxTransSamplesInCol];
	float_point *pfSamplesForARowInSubMatrix;
//	pfSamplesForARowInSubMatrix = new float_point[nMaxSamplesInRow];

	//compute a few rows of Hessian matrix
	float_point *pfDevTransSamples;
	float_point *pfDevNumofHessianRows;
	float_point *pfDevTotalSamples;
	float_point *pfDevSelfDot;
	checkCudaErrors(cudaMalloc((void**)&pfDevTransSamples, sizeof(float_point) * nMaxTransSamplesInCol));
	checkCudaErrors(cudaMalloc((void**)&pfDevTotalSamples, sizeof(float_point) * nMaxSamplesInRow));
	checkCudaErrors(cudaMalloc((void**)&pfDevNumofHessianRows, sizeof(float_point) * lMaxSubMatrixSize));
	checkCudaErrors(cudaMalloc((void**)&pfDevSelfDot, sizeof(float_point) * m_nTotalNumofInstance));
	checkCudaErrors(cudaMemcpy(pfDevSelfDot, pfSelfDot, sizeof(float_point) * m_nTotalNumofInstance, cudaMemcpyHostToDevice));

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
			cout << ".";
			cout.flush();
			//allocate memory for this sub matrix
			long int nHessianSubMatrixSpace = nSubMatrixRow * nSubMatrixCol;
			memset(pfSubMatrix, 0, sizeof(float_point) * nHessianSubMatrixSpace);


			//get the copies of sample data for computing sub matrix

			//sample for sub matrix rows

			/*for(int d = 0; d < m_nNumofDimensions; d++)
			{
				memcpy(pfSamplesForARowInSubMatrix + d * nSubMatrixRow,
						pfTotalSamples + iPartofCol * nAveLenofSubCol + d * m_nTotalNumofSamples,
					   sizeof(float_point) * nSubMatrixRow);
			}
			pfTransSamplesForAColInSubMatrix = pfTransSamples + jPartofRow * nAveLenofSubRow * m_nNumofDimensions;*/
			pfSamplesForARowInSubMatrix = pfTotalSamples + (long int)iPartofCol * nAveLenofSubCol * m_nNumofDimensions;
			for(int d = 0; d < m_nNumofDimensions; d++)
			{
				//for(int k = 0; k < nSubMatrixCol; k++)
				//{
					//pfTransSamplesForAColInSubMatrix[k + d * nSubMatrixCol] =
					//pfTransSamples[k + jPartofRow * nAveLenofSubRow + d * m_nTotalNumofSamples];
				long int nSampleIndex = (long int)jPartofRow * nAveLenofSubRow + (long int)d * m_nTotalNumofInstance;
				long int nIndexForSub = (long int)d * nSubMatrixCol;
					memcpy(pfTransSamplesForAColInSubMatrix + nIndexForSub,
						   pfTransSamples + nSampleIndex, sizeof(float_point) * nSubMatrixCol);
				//}
			}

			long int nSpaceForSamplesInRow = (long int)m_nNumofDimensions * nSubMatrixRow;
			long int nSpaceForSamplesInCol = (long int)m_nNumofDimensions * nSubMatrixCol;
			checkCudaErrors(cudaMemcpy(pfDevTotalSamples, pfSamplesForARowInSubMatrix,
							sizeof(float_point) * nSpaceForSamplesInRow, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(pfDevTransSamples, pfTransSamplesForAColInSubMatrix,
							sizeof(float_point) * nSpaceForSamplesInCol, cudaMemcpyHostToDevice));
//
			//compute the value of the sub matrix
			int nStartRow = iPartofCol * nAveLenofSubCol;
			int nStartCol = jPartofRow * nAveLenofSubRow;
			ComputeSubHessianMatrix(pfDevTotalSamples, pfDevTransSamples, pfDevSelfDot,
									pfDevNumofHessianRows, nSubMatrixCol, nSubMatrixRow,
									nStartRow, nStartCol);

			int nHessianRowsSpace = nSubMatrixRow * nSubMatrixCol;
			checkCudaErrors(cudaMemcpy(pfSubMatrix, pfDevNumofHessianRows,
							sizeof(float_point) * nHessianRowsSpace, cudaMemcpyDeviceToHost));

			//store the sub matrix
			timeval t3, t4;
			float_point elapsedWritingTime;
			gettimeofday(&t3, NULL);
			long lColStartPos = jPartofRow * nAveLenofSubRow;
			//copy the host memory
			if((iPartofCol * nAveLenofSubCol + nSubMatrixRow) <= m_nNumofCachedHessianRow)//the sub matrix should be stored in RAM
			{
				//cout << "copying to host " << lColStartPos << endl;
				for(int k = 0; k < nSubMatrixRow; k++)
				{
					long long lPosInHessian =  (long long)(iPartofCol * nAveLenofSubCol + k) * m_nTotalNumofInstance + lColStartPos;
					long lPosInSubMatrix = k * nSubMatrixCol;
					memcpy(m_pfHessianRowsInHostMem + lPosInHessian, pfSubMatrix + lPosInSubMatrix, sizeof(float_point) * nSubMatrixCol);
				}
			}
			else
			{
				//copy a part of the last row that can fit in host memory
				int nNumofRowsStoredInHost = 0;
				if(iPartofCol * nAveLenofSubCol < m_nNumofCachedHessianRow)
				{
					nNumofRowsStoredInHost = m_nNumofCachedHessianRow - iPartofCol * nAveLenofSubCol;
					//cout << "copying to host " << lColStartPos << endl;
					for(int k = 0; k < nNumofRowsStoredInHost; k++)
					{
						long long lPosInHessian =  (long long)(iPartofCol * nAveLenofSubCol + k) * m_nTotalNumofInstance + lColStartPos;
						long lPosInSubMatrix = k * nSubMatrixCol;
						memcpy(m_pfHessianRowsInHostMem + lPosInHessian, pfSubMatrix + lPosInSubMatrix, sizeof(float_point) * nSubMatrixCol);
					}
				}
//2/4
/*delete[] pfSubMatrix;
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
return true;
*/
				int nNumofRowsToWrite = nSubMatrixRow - nNumofRowsStoredInHost;
				//cout << "writing " << nNumofRowsToWrite << " Hessian sub rows" << endl;
				//the results of this function are: 1. write rows to file; 2. return the index of (start pos of) the rows
				long long lUnstoredStartPos =  (long long)nNumofRowsStoredInHost * nSubMatrixCol;
				//hessian sub matrix info
				SubMatrix subMatrix;
				subMatrix.nColIndex = lColStartPos;
				subMatrix.nColSize = nSubMatrixCol;
				subMatrix.nRowIndex = iPartofCol * nAveLenofSubCol + nNumofRowsStoredInHost;
				//update row index in the file on ssd, as only part of the hessian matrix is stored in file
				subMatrix.nRowIndex -= m_nNumofCachedHessianRow;
				assert(subMatrix.nRowIndex >= 0);

				subMatrix.nRowSize = nNumofRowsToWrite;
//3/4
//bool bWriteRows = true;
bool bWriteRows = WriteHessianRows(writeOut, pfSubMatrix + lUnstoredStartPos, subMatrix);

				if(bWriteRows == false)
				{
					cerr << "error in writing Hessian Rows" << endl;
					bReturn = false;
					break;
				}
				gettimeofday(&t4, NULL);
				elapsedWritingTime = (t4.tv_sec - t3.tv_sec) * 1000.0;
				elapsedWritingTime += (t4.tv_usec - t3.tv_usec) / 1000.0;
				//cout << "storing time " << elapsedWritingTime << " ms.\n";
			}//end store sub matrix to file
			//cout << "one sub matrix stored" << endl;

		}//for each part of a row
	}//for each part of a column

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
//4/4
fclose(writeOut);

	return bReturn;
}
