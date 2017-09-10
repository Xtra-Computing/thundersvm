
#ifndef HOSTSTORAGEMANAGER_H_
#define HOSTSTORAGEMANAGER_H_

class HostStorageManager
{
protected:
	long long m_nFreeMemInFloat;
	HostStorageManager();
	static HostStorageManager *manager;

public:
	static HostStorageManager* getManager();
	virtual ~HostStorageManager();

	int RowInRAM(int, int, int);
};



#endif /* HOSTSTORAGEMANAGER_H_ */
