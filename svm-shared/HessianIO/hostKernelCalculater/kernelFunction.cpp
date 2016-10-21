/*
 * kernelFunction.cpp
 *
 *  Created on: 30/10/2015
 *      Author: Zeyi Wen
 */

#include "kernelFunction.h"

/**
 * @brief: compute dot product of two vectors
 */
float KernelFunction::dotProduct(vector<float> &ins1, vector<float> &ins2)
{
	int numofDim = ins1.size();
	float sum = 0.0;
	for(int i = 0; i < numofDim; i++)
	{
		sum += (ins1[i] * ins2[i]);
	}

	return sum;
}

/*
 * @brief: compute square of a vector
 */
float KernelFunction::square(vector<float> &ins)
{
	float sum = dotProduct(ins, ins);
	return sum;
}

