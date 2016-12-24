/*
 * baseSMO.cu
 *  @brief: definition of some sharable functions of smo solver
 *  Created on: 24 Dec 2016
 *      Author: Zeyi Wen
 */

#include "baseSMO.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

/**
 * @brief: select the first instance in SMO
 */
void BaseSMO::SelectFirst(int numTrainingInstance, float_point CforPositive)
{
	GetBlockMinYiGValue<<<dimGridThinThread, BLOCK_SIZE>>>(devYiGValue, devAlpha, devLabel, CforPositive,
														   numTrainingInstance, devBlockMin, devBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMin, devBlockMinGlobalKey, numofBlock, devYiGValue, NULL, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
	IdofInstanceOne = (int)hostBuffer[0];

	devHessianInstanceRow1 = ObtainRow(numTrainingInstance);
	/*
	devHessianInstanceRow1 = GetHessianRow(numTrainingInstance, IdofInstanceOne);
	//lock cached entry for the sample one, in case it is replaced by sample two
	m_pGPUCache->LockCacheEntry(IdofInstanceOne);
	*/
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
					 numofBlock, devYiGValue, devHessianInstanceRow1, devBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMinYiGValue, numofBlock, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
}
