/*
 * hostHessian.h
 *
 *  Created on: 29/10/2015
 *      Author: Zeyi Wen
 */

#ifndef HOSTHESSIAN_H_
#define HOSTHESSIAN_H_

//self define header files
#include "hostKernelCalculater/kernelFunction.h"
#include "../hessianSubMatrix.h"
#include "baseHessian.h"
#include <iostream>
#include <vector>
using std::string;
using std::vector;

class HostHessian: public BaseHessian
{
public:
	static KernelFunction *m_pKernelFunction;

public:
	HostHessian(KernelFunction *pCalculater){m_pKernelFunction = pCalculater;}
	HostHessian(){}
	virtual ~HostHessian(){}

	virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
							  	   vector<vector<real> > &v_v_DocVector);
};


#endif /* HOSTHESSIAN_H_ */
