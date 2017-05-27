/*
 * parHessianIO.cpp
 *
 *  Created on: 26/11/2013
 *      Author: Zeyi Wen
 */

#include "parAccessor.h"
#include "../constant.h"

int ParAccessor::m_nValueSize = 0;
int ParAccessor::m_nPageSize = 0;
int ParAccessor::m_nBlockSize = 0;
long long ParAccessor::m_nOffset = 0;

int ParAccessor::m_nPageCapacity = 0;	//the number of values that a page can store
int ParAccessor::m_nPagesForARow = 0;	//the number of pages to store a row

int ParAccessor::m_nNumofThread = 0;		//the number of threads to read a row
FILE **ParAccessor::m_pFileReadIn = NULL;
ThreadParameter *ParAccessor::m_pThreadArg = NULL;
bool ParAccessor::m_isFirst = false;
real* ParAccessor::pfHessianFullRow = NULL;

using std::ios;
/*
 * constructor
 */
ParAccessor::ParAccessor()
{
	m_nNumofThread = 4;
	m_nValueSize = 4;
	m_nPageSize = 1024*16;
	m_nBlockSize = 5;//8;
	m_nOffset = 8 * 1024;
	m_isFirst = true;

	m_pFileReadIn = new FILE*[m_nNumofThread];
	m_pThreadArg = new ThreadParameter[m_nNumofThread];
	for(int i = 0; i < m_nNumofThread; i++)
	{
		m_pThreadArg[i].nThreadId = i;
	}

	m_nPageCapacity = m_nPageSize;
	m_nPagesForARow = (m_nTotalNumofInstance + m_nPageCapacity - 1)/ m_nPageCapacity;

	pfHessianFullRow = new real[m_nTotalNumofInstance];
}

/*
 * @brief: write Hessian rows to file
 * @param: writeOut: file stream to write to
 * @param: pfHessianRows: hessian rows to be written
 */
bool ParAccessor::WriteHessianRows(FILE *&writeOut, real *pfHessianRows, SubMatrix &subMatrix)
{
	if(writeOut == NULL || pfHessianRows == NULL || subMatrix.isValid() == false)
	{
		cerr << "error in WriteHessianRows: invalid param" << endl;
		if(pfHessianRows == NULL)
		{
			cerr << "pfHessianRows is NULL" << endl;
		}
		else
		{
			cerr << "writeOut is not open" << endl;
		}
		exit(1);
	}

	//write content to file
	int nNumofRow = subMatrix.nRowSize;
	int nNumofCol = subMatrix.nColSize;
	for(int i = 0; i < nNumofRow; i++)
	{
		long nSSDGlobalRowId = subMatrix.nRowIndex + i;						//row global index
		long long nStartBlockId = (long long)(nSSDGlobalRowId / m_nBlockSize) * m_nPagesForARow;	//start block id
		int nPageId = nSSDGlobalRowId % m_nBlockSize;//page id

		for(int j = 0; j < m_nPagesForARow; j++)//write out page by page
		{
			//compute position in the Hessian matrix
			//long long nPosInHessian = ((long long)m_nBlockSize * (nStartBlockId + j) + nPageId) * (long long)m_nPageSize * m_nValueSize;
			long long nPosInHessian = ((nStartBlockId + j)* (m_nBlockSize * m_nPageSize + m_nOffset) + nPageId * m_nPageSize) * m_nValueSize;
			assert(nPosInHessian >= 0);
			//writeOut.seekp(nPosInHessian, ios::beg);
			fseek(writeOut, nPosInHessian, SEEK_SET);
			assert(ftell(writeOut) != -1);

			long long nMemPos = (long long)i * nNumofCol + (long long)j * m_nPageSize;
			real *pStartPos = pfHessianRows + nMemPos;
			int nNumofRemainingValue = nNumofCol - (j * m_nPageSize);
			if(nNumofRemainingValue >= m_nPageSize)
			{
				//writeOut.write((char*)pStartPos, m_nPageSize * m_nValueSize);//write a page
				fwrite((char*)pStartPos, sizeof(char), m_nPageSize * m_nValueSize, writeOut);//write a page
				if(nNumofRemainingValue == m_nPageSize)
					break;
			}
			else if(nNumofRemainingValue > 0)
			{
				//writeOut.write((char*)pStartPos, nNumofRemainingValue * m_nValueSize);//write the remaining
				fwrite((char*)pStartPos, sizeof(char), nNumofRemainingValue * m_nValueSize, writeOut);
				break;
			}
			else
			{
				cerr << "error in writting parallel accessible matrix" << endl;
				exit(0);
			}
			//writeOut.clear();
		}

		/*if(i == 0){
			fclose(writeOut);
		float_point *pfHessianRow = new float_point[nNumofCol];
		FILE *readIn = fopen(HESSIAN_FILE, "rb");
		m_nRowStartPos1 = 0;
		m_nRowEndPos1 = 53500;
		m_nRowStartPos2 = -1;
		nGlobalRowId += m_nNumofCachedHessianRow;
		ReadHessianRow(readIn, nGlobalRowId, pfHessianRow);
		for(int k = 0; k < nNumofCol; k++)
		{
			if(pfHessianRow[k] != pfHessianRows[i * nNumofCol + k])
				cout << pfHessianRow[k] << " and " << pfHessianRows[i * nNumofCol + k] << endl;
		}
		}*/
	}

	return true;
}

/*
 * @brief: read a sub row of Hessian matrix. This function is used during training & prediction
 * @param: pfHessianRow: a sub row of Hessian Matrix (output of this function)
 */
bool ParAccessor::ReadHessianRow(FILE *&readIn, const int &nRowIdInSSD, real *pfHessianRow)
{
	timespec time1, time2;
	clock_gettime(CLOCK_REALTIME, &time1);
	if(m_isFirst)
	{
		for(int i = 0; i < m_nNumofThread; i++)
		{
			m_pFileReadIn[i] = fopen(HESSIAN_FILE, "rb");
		}
		m_isFirst = false;
	}

	//compute the index of row in the hessian file
	int nSSDRowIndex = nRowIdInSSD;
	assert(nSSDRowIndex >= 0);

	long long nStartBlockId = (long long)(nSSDRowIndex / m_nBlockSize) * m_nPagesForARow;
	int nPageId = nSSDRowIndex % m_nBlockSize;

//	if(m_nRowStartPos1 != -1)
//	{
		//////
		pthread_t *m_pFileReader = new pthread_t[m_nNumofThread];
		for(int t = 0; t < m_nNumofThread; t++)
		{
			m_pThreadArg[t].nRowId = nSSDRowIndex;
			m_pThreadArg[t].pfHessianRow = pfHessianRow;
			pthread_create(&m_pFileReader[t], NULL, ReadRow, (void*)&m_pThreadArg[t]);
		}
		for(int t = 0; t < m_nNumofThread; t++)
			pthread_join(m_pFileReader[t], NULL);

		//construct a subrow
		int nSizeofFirstPart = m_nRowEndPos1 - m_nRowStartPos1 + 1;
		if(m_nRowStartPos1 != -1)
			memcpy(pfHessianRow, pfHessianFullRow, nSizeofFirstPart * sizeof(real));
		else
			nSizeofFirstPart = 0;

		int nSizeofSecondPart = m_nRowEndPos2 - m_nRowStartPos2 + 1;
		if(m_nRowStartPos2 != -1)
			memcpy(pfHessianRow + nSizeofFirstPart, pfHessianFullRow + m_nRowStartPos2, nSizeofSecondPart * sizeof(real));

		//////
		/*for(int i = 0; i < m_nPagesForARow; i++)
		{
			long long nRowPos = (long long)m_nBlockSize * (nStartBlockId + i)  + nPageId;
			 nRowPos *= (long long)m_nPageSize;
			float_point *pfStartPos = pfHessianRow + i * m_nPageSize;

			int nNumofRemainingValue = nNumofValueFirstPart - i * m_nPageSize;
			if(nNumofRemainingValue >= m_nPageSize)
			{
				m_fileOps->ReadPartOfRowFromFile(readIn, pfStartPos, m_nPageSize, nRowPos);
				if(nNumofRemainingValue == m_nPageSize)
					break;
			}
			else if(nNumofRemainingValue > 0)
			{
				int nNumofValueToRead = nNumofRemainingValue;
				m_fileOps->ReadPartOfRowFromFile(readIn, pfStartPos, nNumofValueToRead, nRowPos);
				break;
			}
			else
			{
				cerr << "parallel read a hessian row error" << endl;
				exit(0);
			}
		}*/
		/*for(int j = 0; j < nNumofValueFirstPart; j++)
		{
			if(pfHessianRow[j] != 0)
			{
				cout << "hi" << endl;
			}
		}*/

/*	}

	int nNumofValueSecondPart = 0;
	if(m_nRowStartPos2 != -1)
	{
		cerr << "Oops, haven't implemented parallel read for cross-validation" << endl;
		exit(0);
		nNumofValueSecondPart = m_nRowEndPos2 - m_nRowStartPos2 + 1;
		long long nIndexofFirstElement = (long long)nSSDRowIndex * m_nTotalNumofInstance + m_nRowStartPos2;
		//cout << " index of row " << nIndexofRow << endl;
		//m_fileOps->ReadPartOfRowFromFile(readIn, pfHessianRow + nNumofValueFirstPart, nNumofValueSecondPart, nIndexofFirstElement);
	}
*/
/*	clock_gettime(CLOCK_REALTIME, &time2);
	lIO_counter++;
	long lTemp = ((time2.tv_sec - time1.tv_sec) * 1e9 + (time2.tv_nsec - time1.tv_nsec));
	//if(lTemp > 0)
	{
		lIO_timer += lTemp;
	}
*/
	return true;
}


void *ParAccessor::ReadRow(void *pThreadParameter)
{
	ThreadParameter *pTemp = (ThreadParameter*)pThreadParameter;

	int nRowIndexInFile = pTemp->nRowId;
	real *pfHessianRow = pTemp->pfHessianRow;
	int nTid = pTemp->nThreadId;


	long nStartBlockId = (nRowIndexInFile / m_nBlockSize) * m_nPagesForARow;
	int nPageId = nRowIndexInFile % m_nBlockSize;
	int nNumofValueFirstPart = m_nTotalNumofInstance;//m_nRowEndPos1 - m_nRowStartPos1 + 1;
	for(int i = nTid; i < m_nPagesForARow; i+=m_nNumofThread)
	{
		//long long nRowPos = ((nStartBlockId + i) * (long long)m_nBlockSize + nPageId) * (long long)m_nPageSize;
		long long nRowPos = (nStartBlockId + i) * ((long long)m_nBlockSize * m_nPageSize + m_nOffset) + nPageId * (long long)m_nPageSize;
		real *pfStartPos = pfHessianRow + i * m_nPageSize;

		int nNumofRemainingValue = nNumofValueFirstPart - i * m_nPageSize;
		if(nNumofRemainingValue >= m_nPageSize)
		{
			CFileOps::ReadPartOfRowFromFile(m_pFileReadIn[nTid], pfStartPos, m_nPageSize, nRowPos);
			if(nNumofRemainingValue == m_nPageSize)
				break;
		}
		else if(nNumofRemainingValue > 0)
		{
			int nNumofValueToRead = nNumofRemainingValue;
			CFileOps::ReadPartOfRowFromFile(m_pFileReadIn[nTid], pfStartPos, nNumofValueToRead, nRowPos);
			break;
		}
		else
		{
			cerr << "parallel read a hessian row error" << endl;
			exit(0);
		}
	}

	return NULL;
}
