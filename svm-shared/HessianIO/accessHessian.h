/*
 * accessHessian.h
 *
 *  Created on: 30/10/2015
 *      Author: Zeyi Wen
 */

#ifndef ACCESSHESSIAN_H_
#define ACCESSHESSIAN_H_

#include "../hessianSubMatrix.h"
#include "../fileOps.h"
#include <iostream>
#include <vector>
#include<cstring>
#include<cstdlib>
using std::string;
using std::vector;

class HessianAccessor
{
public:
	static int m_nTotalNumofInstance;

	//for Hessian operation in n-fold-cross-validation
	static int m_nRowStartPos1;	//index of the fisrt part of samples
	static int m_nRowEndPos1;		//index of the end of the first part of samples (include the last sample)
	static int m_nRowStartPos2;	//index of the second part of samplels
	static int m_nRowEndPos2;		//index of the end of the second part of samples (include the last sample)

public:
	HessianAccessor(){}
	virtual ~HessianAccessor(){}

	void SetInvolveData(const int &nStart1, const int &nEnd1, const int &nStart2, const int &nEnd2);
	virtual bool WriteHessianRows(FILE *&writeOut, float *pfHessianRows, SubMatrix &subMatrix) = 0;
	virtual bool ReadHessianRow(FILE *&readIn, const int &nRowIdInSSD, float *pfHessianRow) = 0;
};

#endif /* ACCESSHESSIAN_H_ */
