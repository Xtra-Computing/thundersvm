/*
 * storageManager.h
 *
 *  Created on: 15/10/2015
 *      Author: Zeyi Wen
 */

#ifndef STORAGEMANAGER_H_
#define STORAGEMANAGER_H_

#include "gpu_global_utility.h"

class StorageManager
{
private:
	long long m_nMaxNumofFloatPointInGPU;
	long long m_nFreeMemInFloat;
	StorageManager();
	static StorageManager *manager;

public:
	static StorageManager* getManager();
	~StorageManager();
	int RowInRAM(int, int, int);
	int PartOfRow(int, int);
	int PartOfCol(int, int, int);

	int RowInGPUCache(int, int);

	void ReleaseModel(svm_model&);

private:
	long long GetFreeGPUMem();
};


#endif /* STORAGEMANAGER_H_ */
