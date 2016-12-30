/*
 * baseSMO.cu
 *  @brief: definition of some sharable functions of smo solver
 *  Created on: 24 Dec 2016
 *      Author: Zeyi Wen
 */

#include "baseSMO.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include "smoGPUHelper.h"

/**
 * @brief: initialise some variables of smo solver
 */
void BaseSMO::InitSolver(int nNumofTrainingIns)
{
	numOfBlock = Ceil(nNumofTrainingIns, BLOCK_SIZE);

	//configure cuda kernel
	gridSize = dim3(numOfBlock > NUM_OF_BLOCK ? NUM_OF_BLOCK : numOfBlock, Ceil(numOfBlock, NUM_OF_BLOCK));

	//allocate device memory
	checkCudaErrors(cudaMalloc((void**)&devBlockMin, sizeof(float_point) * numOfBlock));
	checkCudaErrors(cudaMalloc((void**)&devBlockMinGlobalKey, sizeof(int) * numOfBlock));
	//for getting maximum low G value
	checkCudaErrors(cudaMalloc((void**)&devBlockMinYiGValue, sizeof(float_point) * numOfBlock));

	checkCudaErrors(cudaMalloc((void**)&devMinValue, sizeof(float_point)));
	checkCudaErrors(cudaMalloc((void**)&devMinKey, sizeof(int)));

	checkCudaErrors(cudaMallocHost((void **) &hostBuffer, sizeof(float_point) * 5));
	checkCudaErrors(cudaMalloc((void**)&devBuffer, sizeof(float_point) * 5));//only need 4 float_points
}

/**
 * @brief: release solver memory
 */
void BaseSMO::DeInitSolver()
{
    checkCudaErrors(cudaFree(devBlockMin));
    checkCudaErrors(cudaFree(devBlockMinGlobalKey));
    checkCudaErrors(cudaFree(devBlockMinYiGValue));
    checkCudaErrors(cudaFree(devMinValue));
    checkCudaErrors(cudaFree(devMinKey));
    checkCudaErrors(cudaFree(devBuffer));
    checkCudaErrors(cudaFreeHost(hostBuffer));
}

/**
 * @brief: select the first instance in SMO
 */
void BaseSMO::SelectFirst(int numTrainingInstance, float_point CforPositive)
{
	GetBlockMinYiGValue<<<gridSize, BLOCK_SIZE>>>(devYiGValue, devAlpha, devLabel, CforPositive,
														   numTrainingInstance, devBlockMin, devBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMin, devBlockMinGlobalKey, numOfBlock, devYiGValue, NULL, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
	IdofInstanceOne = (int)hostBuffer[0];

	devHessianInstanceRow1 = ObtainRow(numTrainingInstance);
}

/**
 * @breif: select the second instance in SMO
 */
void BaseSMO::SelectSecond(int numTrainingInstance, float_point CforNegative)
{
	float_point fUpSelfKernelValue = 0;
	fUpSelfKernelValue = hessianDiag[IdofInstanceOne];

	//for selecting the second instance
	float_point fMinValue;
	fMinValue = hostBuffer[1];
	upValue = -fMinValue;

	//get block level min (-b_ij*b_ij/a_ij)
	GetBlockMinLowValue<<<gridSize, BLOCK_SIZE>>>
						   (devYiGValue, devAlpha, devLabel, CforNegative, numTrainingInstance, devHessianDiag,
							devHessianInstanceRow1, upValue, fUpSelfKernelValue, devBlockMin, devBlockMinGlobalKey,
							devBlockMinYiGValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE>>>
					(devBlockMin, devBlockMinGlobalKey,
					 numOfBlock, devYiGValue, devHessianInstanceRow1, devBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMinYiGValue, numOfBlock, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
}

/*
 * @brief: update the optimality indicator
 */
void BaseSMO::UpdateYiGValue(int numTrainingInstance, float_point fY1AlphaDiff, float_point fY2AlphaDiff)
{
    float_point fAlpha1 = alpha[IdofInstanceOne];
    float_point fAlpha2 = alpha[IdofInstanceTwo];
    //update yiFvalue
    //copy new alpha values to device
    hostBuffer[0] = IdofInstanceOne;
    hostBuffer[1] = fAlpha1;
    hostBuffer[2] = IdofInstanceTwo;
    hostBuffer[3] = fAlpha2;
    checkCudaErrors(cudaMemcpy(devBuffer, hostBuffer, sizeof(float_point) * 4, cudaMemcpyHostToDevice));
    UpdateYiFValueKernel <<< gridSize, BLOCK_SIZE >>> (devAlpha, devBuffer, devYiGValue,
            devHessianInstanceRow1, devHessianInstanceRow2,
            fY1AlphaDiff, fY2AlphaDiff, numTrainingInstance);
}
