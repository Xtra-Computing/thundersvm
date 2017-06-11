/*
 * hostHessian.cpp
 *
 *  Created on: 29/10/2015
 *      Author: Zeyi Wen
 */

#ifndef HOSTHESSIAN_CPP_
#define HOSTHESSIAN_CPP_

#include <sys/time.h>
#include "hostHessian.h"
#include "../hostStorageManager.h"

KernelFunction *HostHessian::m_pKernelFunction = NULL;

/*
 * @brief: compute Hessian matrix, write Hessian matrix and Hessian diagonal to two files respectively
 * @param: strHessianMatrixFileName: file name of a file storing hessian matrix (which serves as an output of this function)
 * @param: strDiagHessianFileName:	 file name of a file storing diagonal of hessian matrix
 * @param: v_v_DocVector: document vectors of all the training samples
 */

bool HostHessian::PrecomputeHessian(const string &strHessianMatrixFileName,
								 	const string &strDiagHessianFileName,
								 	vector<vector<real> > &v_v_DocVector)
{
	bool bReturn = true;

	m_nTotalNumofInstance = v_v_DocVector.size();
	m_nNumofDim = (v_v_DocVector.front()).size();

	//compute the minimum number of sub matrices that are required to calculate the whole Hessian matrix
	HostStorageManager *manager = HostStorageManager::getManager();
	int nBatchRow = manager->RowInRAM(m_nNumofDim, m_nTotalNumofInstance, m_nTotalNumofInstance);
	//some memory has been used to store a number of rows in RAM
	nBatchRow -= m_nNumofCachedHessianRow;
	if(nBatchRow <= 0)
		nBatchRow = m_nNumofDim;//at least we can compute m_nNumofDim number of rows, due to the conservative estimation in RowInRAM

	pHessianFile = fopen(strHessianMatrixFileName.c_str(), "wb");
	if(pHessianFile == NULL)
	{
		cout << "open " << strHessianMatrixFileName << " failed" << endl;
		exit(0);
	}

	timeval t1, t2;
	double elapsedTime;
	gettimeofday(&t1, NULL);

	assert(m_pKernelFunction != NULL);
	real *pRow = new real[m_nTotalNumofInstance * nBatchRow];
	for(int i = 0; i < m_nTotalNumofInstance; i += nBatchRow)
	{

		//number of rows to compute
		int computingRow = nBatchRow;
//		cout << "computing " << computingRow << " rows of the kernel matrix" << endl;
		if(i + nBatchRow > m_nTotalNumofInstance)
			computingRow = m_nTotalNumofInstance - i;

		m_pKernelFunction->ComputeRow(v_v_DocVector, i, computingRow, pRow);

		SubMatrix subMatrix;
		subMatrix.nColIndex = 0;
		subMatrix.nColSize = m_nTotalNumofInstance;
		subMatrix.nRowIndex = i;
		subMatrix.nRowSize = computingRow;
//		cout << "saving the computed rows..." << endl;
		SaveRows(pRow, subMatrix);
	}
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
	cout << elapsedTime << " ms.\n";
	cout.flush();

	fclose(pHessianFile);
	delete[] pRow;
	return bReturn;
}

#endif /* HOSTHESSIAN_CPP_ */
