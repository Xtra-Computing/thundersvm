/*
 * hostStorageManager.cpp
 *
 *  Created on: 28/10/2015
 *      Author: Zeyi Wen
 */

#include "hostStorageManager.h"

#include <sys/sysinfo.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;
using std::cerr;

HostStorageManager *HostStorageManager::manager = NULL;

HostStorageManager::HostStorageManager()
{
	struct sysinfo info;
	sysinfo(&info);
	m_nFreeMemInFloat = (info.freeram / sizeof(float));
	float percent = 0.8;
	m_nFreeMemInFloat *= percent; //use 80% of the memory for caching
	cout << percent * 100 << "% of the available memory to be used, totaling to "
		 << m_nFreeMemInFloat * 4.0/(1024 * 1024 * 1024) << "GB" << endl;
	assert(m_nFreeMemInFloat > 0);
}

HostStorageManager::~HostStorageManager()
{
	if(manager != NULL)delete manager;
}

HostStorageManager* HostStorageManager::getManager()
{
	if(manager == NULL)
		manager = new HostStorageManager();
	return manager;
}


/**
 * @brief: compute the number of rows stored in main memory
 * @nTotalNumofInstance and @nNumofSample may be the same. The parameters of this function may change
 */
int HostStorageManager::RowInRAM(int nNumofDim, int nTotalNumofInstance, int nNumofSample)
{
	long long nFreeMemInFloat = m_nFreeMemInFloat;
	//memory for storing sample data, both original and transposed forms. That's why we use "2" here.
	long long nMemForSamples = (nNumofDim * (long long)nTotalNumofInstance * 2);
	nFreeMemInFloat -= nMemForSamples; //get the number of available memory in the form of number of float
	if(nFreeMemInFloat <= 0)
	{
		cerr << "the size of free host memory is " << nFreeMemInFloat << endl;
		exit(-1);
	}

	long nNumofHessianRow = (nFreeMemInFloat / nNumofSample);
	if (nNumofHessianRow > nNumofSample)
	{
		//if the available memory is available to store the whole hessian matrix
		nNumofHessianRow = nNumofSample;
	}

	/*
	long nRAMForRow = (RAM_SIZE + 22) * 1024;
	nRAMForRow *= 1024;
	nRAMForRow *= 1024;
	nRAMForRow /= sizeof(float_point);
	nNumofHessianRow = (nRAMForRow / nNumofSample);
	if(nNumofHessianRow > nNumofSample)
		nNumofHessianRow = nNumofSample;
*/
	return nNumofHessianRow;
}
