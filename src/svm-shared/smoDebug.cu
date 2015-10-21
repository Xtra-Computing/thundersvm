/***
 * @Author: Zeyi Wen
 * @Date: Oct 15 2015
 * @Brief:  This file has debugging codes for the SMO solver module
 */

#include "smoSolver.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

void CSMOSolver::StoreRow(float_point *pfDevRow, int nLen)
{
	checkCudaErrors(cudaMalloc((void**)&m_pfRow40, sizeof(float_point) * nLen));
	checkCudaErrors(cudaMemcpy(m_pfRow40, pfDevRow, sizeof(float_point) * nLen, cudaMemcpyDeviceToDevice));
}

/**
 * @brief: this function is for testing; print out the content of a row
 */
void CSMOSolver::PrintTenGPUHessianRow(float_point *pfDevRow, int nLen)
{
	float_point *pfTemp = new float_point[nLen];
	checkCudaErrors(cudaMemcpy(pfTemp, pfDevRow, sizeof(float_point) * nLen, cudaMemcpyDeviceToHost));

	int nTotal = 10;
	for(int i = 0; i < nTotal; i++)
	{
		cout << pfTemp[i];
		if(i < nTotal -1)
			cout << " ";
		else
			cout << endl;
	}

	delete[] pfTemp;
}

/**
 * @brief: this function is for testing; print out the content of a row
 */
void CSMOSolver::PrintGPUHessianRow(float_point *pfDevRow, int nLen)
{
	float_point *pfTemp = new float_point[nLen];
	checkCudaErrors(cudaMemcpy(pfTemp, pfDevRow, sizeof(float_point) * nLen, cudaMemcpyDeviceToHost));

	for(int i = 0; i < nLen; i++)
	{
		cout << pfTemp[i];
		if(i < nLen -1)
			cout << " ";
		else
			cout << endl;

		if(i % 1000 == 0 && i != 0)
			cout << endl;
	}

	delete[] pfTemp;
}


int CSMOSolver::CompareTwoGPURow(float_point *pfDevRow1, float_point *pfDevRow2, int nLen)
{
	float_point *pfRow1 = new float_point[nLen];
	checkCudaErrors(cudaMemcpy(pfRow1, pfDevRow1, sizeof(float_point) * nLen, cudaMemcpyDeviceToHost));
	float_point *pfRow2 = new float_point[nLen];
	checkCudaErrors(cudaMemcpy(pfRow2, pfDevRow2, sizeof(float_point) * nLen, cudaMemcpyDeviceToHost));

	int counter = 0;
	for(int i = 0; i < nLen; i++)
	{
		if(pfRow1[i] != pfRow2[i])
			counter++;
	}

	delete[] pfRow1;
	delete[] pfRow2;

	return counter;
}
