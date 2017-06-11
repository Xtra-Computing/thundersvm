/*
 * rowRAM.cpp
 *
 *  Created on: 15/10/2015
 *      Author: Zeyi Wen
 */

#include "storageManager.h"
#include "gpu_global_utility.h"
#include "constant.h"

#include <sys/sysinfo.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <helper_cuda.h>

using std::cout;
using std::endl;
using std::cerr;


/*
 * @brief: get the size of free memory in the form of float point representation
 */
long long StorageManager::GetFreeGPUMem()
{
	size_t nFreeMem, nTotalMem;
	checkCudaErrors(cuMemGetInfo_v2(&nFreeMem, &nTotalMem));
	cout << "GPU has free memory: " << nFreeMem << "; ";
	if(nTotalMem > 0)
		cout << 100.0 * (double)nFreeMem/nTotalMem << "% of the total memory" << endl;
//	long long nMaxNumofFloatPoint = 0.9 * nFreeMem / sizeof(float_point);
	long long nMaxNumofFloatPoint = nTotalMem / sizeof(real);
	return nMaxNumofFloatPoint;
}

StorageManager::StorageManager()
{
	m_nMaxNumofFloatPointInGPU = GetFreeGPUMem();
	assert(m_nMaxNumofFloatPointInGPU > 0);
}

StorageManager::~StorageManager()
{
	if(manager != NULL)delete manager;
}

StorageManager* StorageManager::getManager()
{
    static StorageManager storageManager;
	return &storageManager;
}

/**
 * @brief: compute the part of a row that can be computed by the GPU
 */
int StorageManager::PartOfRow(int nInstance, int nDim)
{
	//compute the minimum number of sub matrices that are required to calculate the whole Hessian matrix
	int nNumofInsInGPU = ((m_nMaxNumofFloatPointInGPU / (2 * nDim)));//2 is for two copies of samples, original and transposed samples

	//compute the number of partitions of a row
	int nNumofPartForARow = Ceil(nInstance, nNumofInsInGPU);
	if(nNumofPartForARow > 1)//if samples cannot all fit in the GPU memory
		nNumofPartForARow = Ceil(nNumofPartForARow, 2);	//divided by 2 because in the extreme case, the row and column share 0 samples.
	while(true)
	{
		//two copies (original and transposed) of samples; row and col, each needs a copy instances
		int nNumofInsForComp = Ceil(nInstance, nNumofPartForARow);
		long long nNumofElementsForIns = nNumofInsForComp * nDim * 2 * 2;
		long long lElementInSubMatrix = nNumofInsForComp * nNumofInsForComp;
		if(m_nMaxNumofFloatPointInGPU < nNumofElementsForIns + lElementInSubMatrix)
		{
			nNumofPartForARow = nNumofPartForARow * 1.2;
		}
		else
			break;
	}

if(nInstance >= 50000)
	nNumofPartForARow = 2;

	return nNumofPartForARow;
}

/*
 * @brief: compute the number of parts of a column that can be computed
 */
int StorageManager::PartOfCol(int nPartsOfARow, int nInstace, int nDim)
{
	int nNumofPartForACol = nPartsOfARow;//divide the matrix into sub matrices. default is the same for row and col
	if(nPartsOfARow == 1)//can compute a hessian row once
	{//check the maximum number of rows for each computation
		//how many we can compute based on GPU memory constrain
		long nNumofFloatARow = nInstace;
		long lRemainingNumofFloat = m_nMaxNumofFloatPointInGPU - (nInstace * nDim * 2);

		long nMaxNumofRowGPU = lRemainingNumofFloat / nNumofFloatARow;
		if(nMaxNumofRowGPU < 0)
		{
			nMaxNumofRowGPU = 1;
		}

		//how many we can compute based on the RAM constraint
		long nNumofRowCPU = m_nFreeMemInFloat / nNumofFloatARow;

		int nMaxNumofRow = (nNumofRowCPU < nMaxNumofRowGPU ? nNumofRowCPU : nMaxNumofRowGPU);
		nNumofPartForACol = Ceil(nInstace, nMaxNumofRow);
	}
	return nNumofPartForACol;
}


int StorageManager::RowInGPUCache(int nNumofTrainingExample, int nNumofInstance)
{
	//initialize cache
//	long nMaxNumofFloatPoint = m_nMaxNumofFloatPointInGPU;//(long(CACHE_SIZE) * 1024 * 1024 / 4) - (2 * nNumofTrainingExample);//GetFreeGPUMem(); use 500MB memory
	long nMaxNumofFloatPoint = (long(CACHE_SIZE) * 1024 * 1024 / 4) - (2 * nNumofTrainingExample);
	//GPU memory stores a Hessian diagonal, and a few rows of Hessian matrix
	int nMaxCacheSize = (nMaxNumofFloatPoint - nNumofInstance) / nNumofInstance;
	//GPU memory can't go to 100%, so we use a ratio here
	int nSizeofCache = nMaxCacheSize * 0.9;
//if(nNumofInstance == 50000)
//	nSizeofCache = 46000;
//if(nNumofInstance == 53500)
//	nSizeofCache = 42000;
	if(nSizeofCache > nNumofInstance)
	{
		nSizeofCache = nNumofInstance;
	}


	cout << " cache size is: " << nSizeofCache << " max: " << nMaxCacheSize << "; percentage of cached samples: "
		 << ((real)100 * nSizeofCache) / nNumofInstance << "%" << endl;

	return nSizeofCache;
}

void StorageManager::ReleaseModel(svm_model &model)
{
	delete[] model.label;
	delete[] model.nSV;
	delete[] model.pnIndexofSV;
	delete[] model.sv_coef[0];
	delete[] model.sv_coef[1];
	delete[] model.sv_coef[2];
	delete[] model.sv_coef;
	delete[] model.rho;
}

