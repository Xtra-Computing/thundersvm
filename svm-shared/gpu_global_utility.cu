/*
 * utility.cpp
 *
 *  Created on: 16/03/2013
 *      Author: zeyi
 */
#include "gpu_global_utility.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>

using std::cout;
using std::endl;

real gfPCost = 4;	//cost for positive samples in training SVM model (i.e., error tolerance)
real gfNCost = 4;	//cost for negative samples in training SVM model
real gfGamma = 1;
int gNTest = 0;
