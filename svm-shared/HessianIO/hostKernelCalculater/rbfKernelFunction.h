/*
 * rbfKernelFunction.h
 *
 *  Created on: 29/10/2015
 *      Author: Zeyi Wen
 */

#ifndef RBFKERNELFUNCTION_H_
#define RBFKERNELFUNCTION_H_

#include "kernelFunction.h"

class RBFKernelFunction: public KernelFunction
{
public:
	RBFKernelFunction(float gamma){m_gamma = gamma;}
	virtual void ComputeRow(vector<vector<float> > &v_v_DocVector, int rowId, int nNumofRow, float *pRow);

private:
	float RBF(vector<float>&, vector<float>&);
	float m_gamma;
};


#endif /* RBFKERNELFUNCTION_H_ */
