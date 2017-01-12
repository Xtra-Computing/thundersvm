/*
 * initCuda.cpp
 *
 *  Created on: 10/12/2014
 *      Author: Zeyi Wen
 */

#include <helper_cuda.h>
#include <cuda.h>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

/**
 * @brief: set the device to use
 */
void UseDevice(int deviceId, CUcontext &context)
{
    CUdevice device;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, deviceId));
    cout << "Using " << prop.name << "; device id is " << deviceId << endl;
    checkCudaErrors(cudaSetDevice(deviceId));
    cuDeviceGet(&device, deviceId);
    cuCtxCreate(&context, CU_CTX_MAP_HOST, device);
    if(!prop.canMapHostMemory)
		fprintf(stderr, "Device %d cannot map host memory!\n", deviceId);
}


/**
 * @brief: initialize CUDA device
 */

bool InitCUDA(CUcontext &context, char gpuType = 'T')
{
    int count;

    checkCudaErrors(cudaGetDeviceCount(&count));
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
        	cout << prop.name << endl;
        	if(prop.name[0] == gpuType)
        	{//choose the prefer device
                UseDevice(i, context);
       			break;
        	}
        }
    }

    cout << i << " v.s. " << count << endl;
    if(i == count)
    {
        cout << "There is no device of \"" << gpuType << "\" series" << endl;
        UseDevice(0, context);
    }

    return true;
}

bool ReleaseCuda(CUcontext &context)
{
	cuCtxDetach(context);
	return true;
}
