/*
 * seqHessianIO.cpp
 *
 *  Created on: 26/11/2013
 *      Author: Zeyi Wen
 */

#include "seqAccessor.h"

/*
 * @brief: write Hessian rows to file
 * @param: writeOut: file stream to write to
 * @param: pfHessianRows: hessian rows to be written
 */
bool SeqAccessor::WriteHessianRows(FILE *&writeOut, real *pfHessianRows, SubMatrix &subMatrix)
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
		long long nRowPos = (long long)(subMatrix.nRowIndex + i) * (long long)m_nTotalNumofInstance;
		/*if(subMatrix.nRowIndex + i == 10)
		{	for(int m = 0; m < m_nTotalNumofSamples; m++)
				cout << pfHessianRows[i * m_nTotalNumofSamples + m] << " ";
			cout << endl;exit(0);
		}*/
		long long nPosInHessian = nRowPos + subMatrix.nColIndex;
		nPosInHessian *= sizeof(real);
		if(nRowPos < 0 || nPosInHessian < 0)
		{
			cerr << "Position to write is negative!" << endl;
			exit(1);
		}
		//cout << "pos in hessian: " << nPosInHessian << " v.s. row pos: " << nRowPos << endl;
		//writeOut.seekp(nPosInHessian, ios::beg);
		//assert(writeOut.fail() == false);
		fseek(writeOut, nPosInHessian, SEEK_SET);
		assert(ftell(writeOut) != -1);

		//writeOut.write((char*)(pfHessianRows + i * nNumofCol), sizeof(float_point) * nNumofCol);
		fwrite((char*)(pfHessianRows + (long long)i * nNumofCol), sizeof(char), sizeof(real) * nNumofCol, writeOut);
	}


	return true;
}

/*
 * @brief: read a sub row of Hessian matrix. This function is used during training & prediction
 * @param: nNumofInvolveElements: the # of elements involved in training. In n-fold-cross-validation,
 *  not always all elements are involved in training
 * @param: pfHessianRow: a sub row of Hessian Matrix (output of this function)
 */
bool SeqAccessor::ReadHessianRow(FILE *&readIn, const int &nRowIdInSSD, real *pfHessianRow)
{
//	timespec time1, time2;
//	clock_gettime(CLOCK_REALTIME, &time1);

	//compute the index of row in the hessian file
	int nRowIndexInFile = nRowIdInSSD;
//	assert(nRowIndexInFile >= 0);

	long long nIndexofFirstElement;
	int nSizeofFirstPart = 0;
	if(m_nRowStartPos1 != -1)
	{
		nSizeofFirstPart = m_nRowEndPos1 - m_nRowStartPos1 + 1;//the size of first part (include the last element of the part)
		nIndexofFirstElement = (long long)nRowIndexInFile * m_nTotalNumofInstance + m_nRowStartPos1;
		//cout << " index of row " << nIndexofRow << endl;
		CFileOps::ReadPartOfRowFromFile(readIn, pfHessianRow, nSizeofFirstPart, nIndexofFirstElement);
	}
	int nSizeofSecondPart = 0;
	if(m_nRowStartPos2 != -1)
	{
		nSizeofSecondPart = m_nRowEndPos2 - m_nRowStartPos2 + 1;
		nIndexofFirstElement = (long long)nRowIndexInFile * m_nTotalNumofInstance + m_nRowStartPos2;
		//cout << " index of row " << nIndexofRow << endl;
		CFileOps::ReadPartOfRowFromFile(readIn, pfHessianRow + nSizeofFirstPart, nSizeofSecondPart, nIndexofFirstElement);
	}

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
