/*
 * seqHessianIO.h
 *
 *  Created on: 26/11/2013
 *      Author: Zeyi Wen
 */

#ifndef SEQHESSIANIO_H_
#define SEQHESSIANIO_H_

#include "accessHessian.h"

class SeqAccessor: public HessianAccessor
{
public:
	SeqAccessor(){}
	virtual ~SeqAccessor(){}

	virtual bool WriteHessianRows(FILE *&writeOut, real *pfHessianRows, SubMatrix &subMatrix);
	virtual bool ReadHessianRow(FILE *&readIn, const int &nIndexofRow, real *pfHessianRow);
};


#endif /* SEQHESSIANIO_H_ */
