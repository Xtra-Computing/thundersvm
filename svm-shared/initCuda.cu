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

    CUdevice device;
    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
        	cout << prop.name << endl;
        	if(prop.name[0] == gpuType && prop.name[1] == 'e')
        	{//prefere to use Tesla card
        		cout << "Using " << prop.name << "; device id is " << i << endl;
        		checkCudaErrors(cudaSetDevice(i));
        		cuDeviceGet(&device, i);
        		cuCtxCreate(&context, CU_CTX_MAP_HOST, device);

    			cudaGetDeviceProperties(&prop, i);
    			if(!prop.canMapHostMemory)
					fprintf(stderr, "Device %d cannot map host memory!\n", i);

    			break;
        	}
            if(prop.major >= 1)
            {
            	cout << "compute capability: " << prop.major << "; " << count << " devices" << endl;
            }
        }
    }

    cout << i << " v.s. " << count << endl;
    if(i == count)
    {
        cudaDeviceProp prop;
        cout << "There is no device of \"" << gpuType << "\" series" << endl;
        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

        cout << "using " << prop.name << endl;
        checkCudaErrors(cudaSetDevice(0));
        cuDeviceGet(&device, i);
		cuCtxCreate(&context, CU_CTX_MAP_HOST, device);
    }

    return true;
}

