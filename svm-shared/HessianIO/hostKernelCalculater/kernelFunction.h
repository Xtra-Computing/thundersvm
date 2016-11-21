/*
 * kernelFunction.h
 *
 *  Created on: 29/10/2015
 *      Author: Zeyi Wen
 */

#ifndef KERNELFUNCTION_H_
#define KERNELFUNCTION_H_

#include <iostream>
#include <vector>
#include "../../gpu_global_utility.h"

using std::string;
using std::vector;

class KernelFunction
{
public:

	virtual void ComputeRow(vector<vector<float> > &v_v_DocVector, int rowId, int nNumofRows, float *pRow) = 0;
	virtual void ComputeSparseRow(vector<vector<svm_node> >&v_v_DocVector, int rowId, int nNumofRow, float* pRow)= 0;

	float dotProduct(vector<float>&, vector<float>&);
	float square(vector<float>&);
};


#endif /* KERNELFUNCTION_H_ */
