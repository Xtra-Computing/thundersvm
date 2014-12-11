/**
 * hessianIO.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef HESSIANIO_H_
#define HESSIANIO_H_

//self define header files
#include "../kernelCalculater/kernelCalculater.h"
#include "../hessianSubMatrix.h"
#include "../fileOps.h"
#include <iostream>
#include <vector>
using std::string;
using std::vector;

extern long lIO_timer;
extern long lIO_counter;
class CHessianIOOps
{
public:
	static int m_nTotalNumofInstance;
	static int m_nNumofDimensions;
	static float_point *m_pfHessianRowsInHostMem;
	static int m_nNumofCachedHessianRow;
	static float_point *m_pfHessianDiag;
	static CKernelCalculater *m_pKernelCalculater;

	int m_nNumofHessianRowsToWrite;	//batch write. Group a few rows of hessian matrix to write at one time
	float_point *m_pfHessianRows;

	//for Hessian operation in n-fold-cross-validation
	static int m_nRowStartPos1;	//index of the fisrt part of samples
	static int m_nRowEndPos1;		//index of the end of the first part of samples (include the last sample)
	int m_nRowStartPos2;	//index of the second part of samplels
	int m_nRowEndPos2;		//index of the end of the second part of samples (include the last sample)

	//object member
	static CFileOps *m_fileOps;

public:
	CHessianIOOps(CKernelCalculater *pCalculater)
	{
		m_pKernelCalculater = pCalculater;
		m_pfHessianRows = NULL;
	}
	CHessianIOOps()
	{
		m_pfHessianRows = NULL;
	}
	virtual ~CHessianIOOps(){}

	//during n-fold-cross-validation, usually part of the samples are involved in training, or prediction
	bool SetInvolveData(const int &nStart1, const int &nEnd1, const int &nStart2, const int &nEnd2);
	//set kernel function for computing Hessian Matrix
	void SetKernelCalculater(CKernelCalculater *pCalculater){m_pKernelCalculater = pCalculater;}

	//read a sub row of Hessian Matrix. This function is used during training & prediction stage.
	virtual bool ReadHessianRow(FILE *&readIn, const int &nIndexofRow, float_point *pfFullHessianRow) = 0;
	virtual bool WriteHessianRows(FILE *&writeOut, float_point *pfHessianRows, SubMatrix &subMatrix) = 0;

	bool ReadHessianRow(boost::interprocess::mapped_region *region, const int &nIndexofRow, const int &nNumofInvolveElements,
						float_point *pfFullHessianRow, int nNumofElementEachRowInCache);

	//read a full row of Hessian Matrix. Getting sub row may need to get the full row of Hessian Matrix
	bool ReadHessianFullRow(FILE *&readIn, const int &nIndexofRow,  int nNumofRowsToRead, float_point *pfFullHessianRow);
	//read a sub row of Hessian Matrix. This function is used during prediction stage, as prediction sub row is consisted of one continous part.
	bool ReadHessianSubRow(FILE *&readIn, const int &nIndexofRow,
						   const int &nStartPos, const int &nEndPos,
						   float_point *pfHessianSubRow);
	//read a few sub rows at once. This function is for initialization of cache
	bool ReadHessianRows(FILE *&readIn, const int &nStartRow, const int &nEndRow, const int &nNumofInvolveElements, float_point *pfHessianRow, int nNumOfElementEachRowInCache);
	//read Hessian diagonal
	bool GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag);
	void ReadDiagFromHessianMatrix();

	bool WriteHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName, vector<vector<float_point> > &v_v_DocVector);

	bool ComputeHessianRows(float_point *pfDevTrainingSamples, float_point *pfDevNumofHessianRows, const int &nStartRow);
	bool ComputeSubHessianMatrix(float_point*, float_point*, float_point*selfDot, float_point*, int, int, int nStartRow, int nStartCol);

	int GetNumofBatchWriteRows();
	bool MapIndexToHessian(int &nIndex);

	//allocate reading buffer
	bool AllocateBuffer(int nNumofRows);
	bool ReleaseBuffer();

protected:
	void ComputeHessianAtOnce(float_point *pfTotalSamples, float_point *pfTransSamples, float_point *pfSelfDot);
	void ComputeSubHessian();
};

#endif /* HESSIANIO_H_ */
