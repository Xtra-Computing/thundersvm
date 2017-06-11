/*
 * hostHessian.cpp
 *
 *  Created on: 28/10/2015
 *      Author: Zeyi Wen
 */


#include "baseHessian.h"
#include <helper_cuda.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include "../gpu_global_utility.h"
#include "../constant.h"
#include "../storageManager.h"

using std::endl;

//initialize the static variables for Hessian Operator
int BaseHessian::m_nTotalNumofInstance = 0;
int BaseHessian::m_nNumofDim = 0;
real* BaseHessian::m_pfHessianRowsInHostMem = NULL;
real* BaseHessian::m_pfHessianDiag = NULL;
int BaseHessian::m_nNumofCachedHessianRow = 0;

int BaseHessian::m_nNumofHessianRowsToWrite = -1;	//batch write. Group a few rows of hessian matrix to write at one time
real *BaseHessian::m_pfHessianRows = NULL;

int BaseHessian::m_nRowStartPos1 = -1;
int BaseHessian::m_nRowEndPos1 = -1;
HessianAccessor *BaseHessian::pAccessor = NULL;
FILE *BaseHessian::pHessianFile = NULL;
int BaseHessian::m_nRowStartPos2 = -1;
int BaseHessian::m_nRowEndPos2 = -1;

/*
 * @brief: set data involved in Hessian Read Operation
 * @param: nStart1: the index of the first part of a row
 * @param: nEnd1: the index of the end of the first part of a row
 * @param: nStart2: the index of the second part of a row
 * @param: nEnd2: the index of the end of the second part of a row
 */
bool BaseHessian::SetInvolveData(const int &nStart1, const int &nEnd1, const int &nStart2, const int &nEnd2)
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
 * @brief: allocate memory for reading content from file
 */
bool BaseHessian::AllocateBuffer(int nNumofRows)
{
	bool bReturn = false;

	if(nNumofRows < 1)
	{
		cerr << "error in hessian ops: allocate buffer failed!" << endl;
		return bReturn;
	}
	bReturn = true;
	m_pfHessianRows = new real[m_nTotalNumofInstance * nNumofRows];

	return bReturn;
}

/*
 * @brief: release buffer from reading hessian rows
 */
bool BaseHessian::ReleaseBuffer()
{
	if(m_pfHessianRows == NULL)
	{
		cerr << "buffer to be released is empty!" << endl;
		return false;
	}
	delete[] m_pfHessianRows;
	return true;
}


void BaseHessian::ReadDiagFromHessianMatrix()
{
	real *hessianRow = new real[m_nTotalNumofInstance];

	FILE *readIn = fopen(HESSIAN_FILE, "rb");
	if(readIn == NULL)
	{
		cerr << "opening file \"" << HESSIAN_FILE << "\" failed" << endl;
		exit(-1);
	}
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

	delete[] hessianRow;
}

bool BaseHessian::MapIndexToHessian(int &nIndex)
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

/*
 * @brief: read one full Hessian row from file
 * @return: true if read the row successfully
 */
bool BaseHessian::ReadHessianFullRow(FILE *&readIn, const int &nIndexofRow, int nNumofRowsToRead, real *pfFullHessianRow)
{
	bool bReturn = false;
	assert(readIn != NULL && nIndexofRow >= 0 && nIndexofRow < m_nTotalNumofInstance);

	//read the whole Hessian row
	bReturn = CFileOps::ReadRowsFromFile(readIn, pfFullHessianRow, m_nTotalNumofInstance, nNumofRowsToRead, nIndexofRow);
	assert(bReturn != false && pfFullHessianRow != NULL);

	return bReturn;
}

/*
 * @brief: read a continuous part of a Hessian row. Note that the last element (nEndPos) is included in the sub row
 * @output: pfHessianSubRow: part of a Hessian row
 */
bool BaseHessian::ReadHessianSubRow(FILE *&readIn, const int &nIndexofRow,
		   	   	   	   	   	   	   	  const int &nStartPos, const int &nEndPos,
		   	   	   	   	   	   	   	  real *pfHessianSubRow)
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
	real *pfTempFullHessianRow = new real[m_nTotalNumofInstance];
	bool bReadRow =	ReadHessianFullRow(readIn, nIndexofRow, 1, pfTempFullHessianRow); //1 means that read one Hessian row
	if(bReadRow == false)
	{
		cerr << "error in ReadHessianRow" << endl;
		delete[] pfTempFullHessianRow;
		return bReturn;
	}

	//get sub row from a full Hessian row
	memcpy(pfHessianSubRow, pfTempFullHessianRow + nStartPos, sizeof(real) * nNumofHessianElements);

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
bool BaseHessian::ReadHessianRows(FILE *&readIn, const int &nStartRow, const int &nEndRow,
									const int &nNumofInvolveElements, real * pfHessianRow, int nNumOfElementEachRowInCache)
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
	real *pfTempFullHessianRow;// = new float_point[m_nTotalNumofSamples];
	for(int i = nStartRow; i <= nEndRow; i++)
	{
		pfTempFullHessianRow = m_pfHessianRows + (i - nStartRow) * m_nTotalNumofInstance;

		//read the first continuous part
		if(m_nRowStartPos1 != -1)
		{
			//first part is added to the end of current Hessian space in main memory
			nHessianEndPos = (i - nStartRow) * nNumOfElementEachRowInCache;//use number of elements each row instead of number of involve elements due to memory alignment
			memcpy(pfHessianRow + nHessianEndPos, pfTempFullHessianRow + m_nRowStartPos1, sizeof(real) * nSizeofFirstPart);
		}
		//read the second continuous part
		if(m_nRowStartPos2 != -1)
		{
			nHessianEndPos = (i - nStartRow) * nNumOfElementEachRowInCache + nSizeofFirstPart;
			memcpy(pfHessianRow + nHessianEndPos, pfTempFullHessianRow + m_nRowStartPos2, sizeof(real) * nSizeofSecondPart);
		}
	}
	//delete[] pfTempHessianRows;
	return bReturn;
}

/**
 * @brief: save a pre-computed sub-matrix (rows) to host memory or SSDs
 * @pfSubMatrix: kernel values to be saved
 * @subMatrix: information about the sub-matrix
 */
void BaseHessian::SaveRows(real *pfSubMatrix, const SubMatrix &subMatrix)
{
	//store the sub matrix
	long lColStartPos = subMatrix.nColIndex;

	//the sub matrix should be stored in RAM
	int nRowId = subMatrix.nRowIndex;
	int nSubMatrixRow = subMatrix.nRowSize;
	int nSubMatrixCol = subMatrix.nColSize;

	if(nRowId + nSubMatrixRow <= m_nNumofCachedHessianRow)
	{
		//copying to host memory
		for(int k = 0; k < nSubMatrixRow; k++)
		{
			long long lPosInHessian =  (long long)(nRowId + k) * m_nTotalNumofInstance + lColStartPos;
			long lPosInSubMatrix = k * nSubMatrixCol;
			memcpy(m_pfHessianRowsInHostMem + lPosInHessian, pfSubMatrix + lPosInSubMatrix, sizeof(real) * nSubMatrixCol);
		}
	}
	else
	{
		//copy a part of the last row that can fit in host memory
		int nNumofRowsStoredInHost = 0;
		if(nRowId < m_nNumofCachedHessianRow)
		{
			nNumofRowsStoredInHost = m_nNumofCachedHessianRow - nRowId;
			//cout << "copying to host " << lColStartPos << endl;
			for(int k = 0; k < nNumofRowsStoredInHost; k++)
			{
				long long lPosInHessian =  (long long)(nRowId + k) * m_nTotalNumofInstance + lColStartPos;
				long lPosInSubMatrix = k * nSubMatrixCol;
				memcpy(m_pfHessianRowsInHostMem + lPosInHessian, pfSubMatrix + lPosInSubMatrix, sizeof(real) * nSubMatrixCol);
			}
		}

		int nNumofRowsToWrite = nSubMatrixRow - nNumofRowsStoredInHost;
		//the results of this function are: 1. write rows to file; 2. return the index of (start pos of) the rows
		long long lUnstoredStartPos =  (long long)nNumofRowsStoredInHost * nSubMatrixCol;

		//hessian sub matrix info
		SubMatrix subTempMatrix;
		subTempMatrix.nColIndex = lColStartPos;
		subTempMatrix.nColSize = nSubMatrixCol;
		subTempMatrix.nRowIndex = nRowId + nNumofRowsStoredInHost;
		//update row index in the file on ssd, as only part of the hessian matrix is stored in file
		subTempMatrix.nRowIndex -= m_nNumofCachedHessianRow;
		assert(subTempMatrix.nRowIndex >= 0);

		subTempMatrix.nRowSize = nNumofRowsToWrite;

		bool bWriteRows = pAccessor->WriteHessianRows(pHessianFile, pfSubMatrix + lUnstoredStartPos, subTempMatrix);

		if(bWriteRows == false)
		{
			cerr << "error in writing Hessian Rows" << endl;
			exit(-1);
		}
	}//end store sub matrix to file
}

/**
 * @brief: read a row from the precomputed kernel matrix
 */
void BaseHessian::ReadRow(int nPosofRowAtHessian, real *pfHessianRow)
{
	memset(pfHessianRow, 0, sizeof(real) * m_nTotalNumofInstance);
	//if the hessian row is in host memory
	if(m_nNumofCachedHessianRow > nPosofRowAtHessian)
	{
		int nSizeofFirstPart = 0;
		if(m_nRowStartPos1 != -1)
		{
			nSizeofFirstPart = m_nRowEndPos1 - m_nRowStartPos1 + 1;//the size of first part (include the last element of the part)
			long long nIndexofFirstElement = (long long)nPosofRowAtHessian * (m_nTotalNumofInstance) + m_nRowStartPos1;
			memcpy(pfHessianRow, m_pfHessianRowsInHostMem + nIndexofFirstElement, nSizeofFirstPart * sizeof(real));
		}
		if(m_nRowStartPos2 != -1)
		{
			int nSizeofSecondPart = m_nRowEndPos2 - m_nRowStartPos2 + 1;
			long long nIndexofFirstElement = (long long)nPosofRowAtHessian * (m_nTotalNumofInstance) + m_nRowStartPos2;
			memcpy(pfHessianRow + nSizeofFirstPart, m_pfHessianRowsInHostMem + nIndexofFirstElement,
				   nSizeofSecondPart * sizeof(real));
		}
	}
	else//the hessian row is in SSD
	{
		int nIndexInSSD = nPosofRowAtHessian - m_nNumofCachedHessianRow;
		pAccessor->ReadHessianRow(pHessianFile, nIndexInSSD, pfHessianRow);
	}
}


void BaseHessian::PrintHessianInfo()
{
	cout << "ins=" << m_nTotalNumofInstance << "\t";
	cout << "dim=" << m_nNumofDim << "\t";
	cout << "ram row=" << m_nNumofCachedHessianRow << "\t";
	cout << "part1_start=" << m_nRowStartPos1 << "\t";
	cout <<	"part1_end=" << m_nRowEndPos1 << "\t";
	cout <<	"part2_start=" << m_nRowStartPos2 << "\t";
	cout << "part2_end=" << m_nRowEndPos2 << endl;
}


/**
 * @brief: kernel matrix precomputation
 */
void BaseHessian::PrecomputeKernelMatrix(vector<vector<real> > &v_vDocVector, BaseHessian *hessianIOOps)
{
	//compute Hessian Matrix
	string strHessianMatrixFileName = HESSIAN_FILE;
	string strDiagHessianFileName = HESSIAN_DIAG_FILE;

	int nNumofSample = v_vDocVector.size();

	//initialize Hessian IO operator
	//CLinearKernel RBF(pfGamma[j]);

	int nNumofRowsOfHessianMatrix = v_vDocVector.size();
	//space of row-index-in-file is for improving reading performace
	BaseHessian::m_nNumofDim = v_vDocVector.front().size();
	BaseHessian::m_nTotalNumofInstance = nNumofRowsOfHessianMatrix;

	StorageManager *manager = StorageManager::getManager();
	int nNumofHessianRow = manager->RowInRAM(BaseHessian::m_nNumofDim, BaseHessian::m_nTotalNumofInstance, nNumofSample);

	cout << nNumofHessianRow << " rows cached in RAM" << endl;
	long long lSizeofCachedHessian = sizeof(real) * (long long)nNumofHessianRow * nNumofSample;


	cout << "numRow " << nNumofHessianRow << "; numIns " << nNumofSample << "; numBytes " << lSizeofCachedHessian << endl;
	if(lSizeofCachedHessian < 0)
	{
		cerr << "locate negative amount of host memory" << endl;
		exit(-1);
	}

	checkCudaErrors(cudaMallocHost((void**)&(BaseHessian::m_pfHessianRowsInHostMem), lSizeofCachedHessian));

	memset(BaseHessian::m_pfHessianRowsInHostMem, 0, lSizeofCachedHessian);
	BaseHessian::m_nNumofCachedHessianRow = nNumofHessianRow;
	BaseHessian::m_pfHessianDiag = new real[hessianIOOps->m_nTotalNumofInstance];
	//hessianIOOps->m_pfHessianDiagTest = new float_point[hessianIOOps->m_nTotalNumofInstance];

	//pre-compute Hessian Matrix and store the result into a file
	cout << "precomputing kernel matrix...";
	cout.flush();

	timeval t1, t2;
	real elapsedTime;
	gettimeofday(&t1, NULL);
	bool bWriteHessian = hessianIOOps->PrecomputeHessian(strHessianMatrixFileName, strDiagHessianFileName, v_vDocVector);
	hessianIOOps->ReadDiagFromHessianMatrix();

	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
	//cout << "Done" << elapsedTime << " ms.\n";
	cout << " Done" << endl;

	if(bWriteHessian == false)
	{
		cerr << "write matrix to file failed" << endl;
		exit(0);
	}
}

