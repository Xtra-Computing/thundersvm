/*
 * rbfKernelFunction.cpp
 *
 *  Created on: 29/10/2015
 *      Author: Zeyi Wen
 */

#include <assert.h>
#include <math.h>
#include <iostream>
#include "rbfKernelFunction.h"

using std::cout;
using std::endl;

void RBFKernelFunction::ComputeRow(vector<vector<float> > &vvDocVector, int rowId, int nNumofRow, float *pRow)
{
	assert(rowId >= 0 && nNumofRow > 0 && pRow != NULL);
	int numofInstance = vvDocVector.size();
	assert(numofInstance > 0);
	for(int j = 0; j < nNumofRow; j++)
	{//compute row of index at rowId
//		cout << "computing " << j << " row" << endl;
		for(int i = 0; i < numofInstance; i++)
		{
			float value_ij = RBF(vvDocVector[j + rowId], vvDocVector[i]);

			pRow[i + j * numofInstance] = value_ij;
		}
	}
}

/*
 * @brief: compute the kernel value of two instances
 */
float RBFKernelFunction::RBF(vector<float> &ins1, vector<float> &ins2)
{
	float value = 0;

	value = exp(-m_gamma * (square(ins1) + square(ins2) - 2*dotProduct(ins1,ins2)));

	return value;
}
