/***
 * @Author: Zeyi Wen
 * @Date: Oct 15 2015
 * @Brief:  This file has debugging codes for the SMO solver module
 */

#include "smoSolver.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>

void CSMOSolver::StoreRow(real *pfDevRow, int nLen)
{
	checkCudaErrors(cudaMalloc((void**)&m_pfRow40, sizeof(real) * nLen));
	checkCudaErrors(cudaMemcpy(m_pfRow40, pfDevRow, sizeof(real) * nLen, cudaMemcpyDeviceToDevice));
}

/**
 * @brief: this function is for testing; print out the content of a row
 */
void CSMOSolver::PrintTenGPUHessianRow(real *pfDevRow, int nLen)
{
	real *pfTemp = new real[nLen];
	checkCudaErrors(cudaMemcpy(pfTemp, pfDevRow, sizeof(real) * nLen, cudaMemcpyDeviceToHost));

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
void CSMOSolver::PrintGPUHessianRow(real *pfDevRow, int nLen)
{
	real *pfTemp = new real[nLen];
	checkCudaErrors(cudaMemcpy(pfTemp, pfDevRow, sizeof(real) * nLen, cudaMemcpyDeviceToHost));

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


int CSMOSolver::CompareTwoGPURow(real *pfDevRow1, real *pfDevRow2, int nLen)
{
	real *pfRow1 = new real[nLen];
	checkCudaErrors(cudaMemcpy(pfRow1, pfDevRow1, sizeof(real) * nLen, cudaMemcpyDeviceToHost));
	real *pfRow2 = new real[nLen];
	checkCudaErrors(cudaMemcpy(pfRow2, pfDevRow2, sizeof(real) * nLen, cudaMemcpyDeviceToHost));

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
