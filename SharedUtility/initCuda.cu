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
 * @brief: get the device with maximum memory
 */
int GetMaxMemDevice(int count){
	int id = 0;
	unsigned int maxMem = 0;
	for(int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			//check memory size
			checkCudaErrors(cudaSetDevice(i));
			size_t nFreeMem, nTotalMem;
			checkCudaErrors(cudaMemGetInfo(&nFreeMem, &nTotalMem));  
			if(nFreeMem > maxMem){
				maxMem = nFreeMem;
				id = i;
			}
        }
    }
	return id;
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

	//use the device with the largest available memory
	int bestId = GetMaxMemDevice(count);
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, bestId));
	if(prop.name[0] == gpuType){
		UseDevice(bestId, context);
		return true;
	}

	//choose the device with the prefer name
    int i;
    for(i = 0; i < count; i++) {
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
