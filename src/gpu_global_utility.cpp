/*
 * utility.cpp
 *
 *  Created on: 16/03/2013
 *      Author: zeyi
 */
#include "gpu_global_utility.h"
#include <cuda.h>



float_point gfPCost = 0.01;	//cost for positive samples in training SVM model (i.e., error tolerance)
float_point gfNCost = 0.01;	//cost for negative samples in training SVM model
float_point gfGamma = 1;
int gNTest = 0;
int gnNumofThread = 0;

/*
 * @brief: get the size of free memory in the form of float point representation
 */
int GetFreeGPUMem()
{
	size_t nFreeMem, nTotalMem;
	cuMemGetInfo(&nFreeMem, &nTotalMem);
	int nMaxNumofFloatPoint = (400 * 1024 * 1024) / sizeof(float_point);
	return nMaxNumofFloatPoint;
}
