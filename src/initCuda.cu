/*
 * initCuda.cpp
 *
 *  Created on: 10/12/2014
 *      Author: Zeyi Wen
 */

#include <helper_cuda.h>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

/**
 * @brief: initialize CUDA device
 */
bool InitCUDA()
{
    int count;

    checkCudaErrors(cudaGetDeviceCount(&count));
    if(count == 0)
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    cudaDeviceProp prop;
    checkCudaErrors(cudaSetDevice(0));
    if(cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
    {
        cout << "using " << prop.name << endl;
    }

	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "cuda error after initCuda" << endl;
		return false;
	}

    return true;
}

