/*
 * parHessianIO.h
 *
 *  Created on: 26/11/2013
 *      Author: Zeyi Wen
 */

#ifndef PARHESSIANIO_H_
#define PARHESSIANIO_H_

#include "accessHessian.h"
#include<cstdlib>
#include<cstring>

struct ThreadParameter
{
	int nRowId;
	int nThreadId;
	real *pfHessianRow;
};

class ParAccessor: public HessianAccessor
{
public:
	static int m_nValueSize;
	static int m_nPageSize;
	static int m_nBlockSize;

	static int m_nPageCapacity;	//the number of values that a page can store
	static int m_nPagesForARow;	//the number of pages to store a row

	static int m_nNumofThread;		//the number of threads to read a row
	static FILE **m_pFileReadIn;
	static ThreadParameter *m_pThreadArg;
	static bool m_isFirst;
	static long long m_nOffset;

	static real *pfHessianFullRow;

public:
	ParAccessor();
	virtual ~ParAccessor(){}

	virtual bool WriteHessianRows(FILE *&writeOut, real *pfHessianRows, SubMatrix &subMatrix);
	virtual bool ReadHessianRow(FILE *&readIn, const int &nIndexofRow, real *pfHessianRow);

private:
	static void *ReadRow(void *pThreadParameter);
};

#endif /* PARHESSIANIO_H_ */
