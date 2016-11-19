/*
 * storageManager.h
 *
 *  Created on: 15/10/2015
 *      Author: Zeyi Wen
 */

#ifndef STORAGEMANAGER_H_
#define STORAGEMANAGER_H_

#include "gpu_global_utility.h"
#include "hostStorageManager.h"

class StorageManager: public HostStorageManager
{
private:
	long long m_nMaxNumofFloatPointInGPU;
//	static StorageManager *manager;
	StorageManager();
	virtual ~StorageManager();
    StorageManager&operator=(const StorageManager&);
	StorageManager(const StorageManager&);
public:
	static StorageManager* getManager();
	int PartOfRow(int, int);
	int PartOfCol(int, int, int);

	int RowInGPUCache(int, int);

	void ReleaseModel(svm_model&);

	long long GetFreeGPUMem();
};


#endif /* STORAGEMANAGER_H_ */
