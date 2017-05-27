/*
 * hostHessian.h
 *
 *  Created on: 28/10/2015
 *      Author: Zeyi Wen
 */

#ifndef BASEHESSIAN_H_
#define BASEHESSIAN_H_

//self define header files
#include "../fileOps.h"
#include "accessHessian.h"
#include <iostream>
#include <vector>
#include<cstring>
#include<cstdlib>
using std::string;
using std::vector;

class BaseHessian
{
public:
	static int m_nTotalNumofInstance;
	static int m_nNumofDim;
	static real *m_pfHessianRowsInHostMem;
	static int m_nNumofCachedHessianRow;
	static real *m_pfHessianDiag;

	static int m_nNumofHessianRowsToWrite;	//batch write. Group a few rows of hessian matrix to write at one time
	static real *m_pfHessianRows;

	//for Hessian operation in n-fold-cross-validation
	static int m_nRowStartPos1;	//index of the fisrt part of samples
	static int m_nRowEndPos1;		//index of the end of the first part of samples (include the last sample)
	static int m_nRowStartPos2;	//index of the second part of samplels
	static int m_nRowEndPos2;		//index of the end of the second part of samples (include the last sample)

	//object member
	static HessianAccessor *pAccessor;
	static FILE *pHessianFile;

public:
	BaseHessian()
	{
		m_pfHessianRows = NULL;
		m_nNumofHessianRowsToWrite = -1;
		m_nRowStartPos2 = -1;
		m_nRowEndPos2 = -1;
	}
	virtual ~BaseHessian(){}

	//during n-fold-cross-validation, usually part of the samples are involved in training, or prediction
	bool SetInvolveData(const int &nStart1, const int &nEnd1, const int &nStart2, const int &nEnd2);
	void SetAccessor(HessianAccessor *accessor){pAccessor = accessor;}


	//read Hessian diagonal
	void ReadDiagFromHessianMatrix();

	bool MapIndexToHessian(int &nIndex);

    virtual //allocate reading buffer
	bool AllocateBuffer(int nNumofRows);

	virtual bool ReleaseBuffer();

	void SaveRows(real *pfSubMatrix, const SubMatrix &subMatrix);

    virtual void ReadRow(int nPosofRowAtHessian, real *pfHessianRow);

	void PrecomputeKernelMatrix(vector<vector<real> > &v_vDocVector, BaseHessian *hessianIOOps);
	virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName, vector<vector<real> > &v_v_DocVector) = 0;
    virtual bool GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, real *pfHessianDiag) {
		return false;
	}

private:
	//read a full row of Hessian Matrix. Getting sub row may need to get the full row of Hessian Matrix
	bool ReadHessianFullRow(FILE *&readIn, const int &nIndexofRow,  int nNumofRowsToRead, real *pfFullHessianRow);
	//read a sub row of Hessian Matrix. This function is used during prediction stage, as prediction sub row is consisted of one continous part.
	bool ReadHessianSubRow(FILE *&readIn, const int &nIndexofRow,
						   const int &nStartPos, const int &nEndPos,
						   real *pfHessianSubRow);
	//read a few sub rows at once. This function is for initialization of cache
	bool ReadHessianRows(FILE *&readIn, const int &nStartRow, const int &nEndRow, const int &nNumofInvolveElements,
						 real *pfHessianRow, int nNumOfElementEachRowInCache);

public:
	static void PrintHessianInfo();
};


#endif /* BASEHESSIAN_H_ */
