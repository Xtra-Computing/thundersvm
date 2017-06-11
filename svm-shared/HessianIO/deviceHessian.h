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
#include "baseHessian.h"
#include <iostream>
#include <vector>
using std::string;
using std::vector;

extern long lIO_timer;
extern long lIO_counter;
class DeviceHessian: public BaseHessian
{
public:
	static CKernelCalculater *m_pKernelCalculater;

public:
	DeviceHessian(CKernelCalculater *pCalculater){m_pKernelCalculater = pCalculater;}
	DeviceHessian(){}
	virtual ~DeviceHessian(){}

	//set kernel function for computing Hessian Matrix
	void SetKernelCalculater(CKernelCalculater *pCalculater){m_pKernelCalculater = pCalculater;}

	virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
							  	   vector<vector<real> > &v_v_DocVector);
	bool GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, real *pfHessianDiag);

protected:
	bool ComputeSubHessianMatrix(real*, real*, real*selfDot, real*, int, int, int nStartRow, int nStartCol);
	int GetNumofBatchWriteRows();
	void ComputeHessianAtOnce(real *pfTotalSamples, real *pfTransSamples, real *pfSelfDot);
};

#endif /* HESSIANIO_H_ */
