/**
 * cache.h
 * Created on: May 28, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 * @brief: this file has classes for caching kernel values
 **/

#ifndef CACHE_H_
#define CACHE_H_

#include <vector>
#include <map>
#include <set>
#include <list>
#include <iostream>
#include "../my_assert.h"
#include <fstream>
#include "../gpu_global_utility.h"

using std::list;
using std::vector;
using std::map;
using std::set;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ios;
using std::string;

/*
 * @brief: three possible values of cache status
 * @NEVER: this entry has never been in cache
 * @EVICTED: the entry has been removed from cache
 * @CACHED: the entry is in cache
 */
enum
{
	NEVER, EVICTED, CACHED
};

/*
 * @brief: abstract cache class
 */
class CCache
{
public:
	int m_nCompulsoryMisses;
	int m_nCapacityMisses;
	int m_nCacheSize;

protected:
	int m_nNumofHits;
	int m_nCacheOccupancy;
	int m_nNumofSamples;

public:
	CCache()
	{
		m_nCacheSize = 0;
		m_nCacheOccupancy = 0;
		m_nNumofHits = 0;
		m_nCompulsoryMisses = 0;
		m_nCapacityMisses = 0;
	}
	virtual ~CCache();

	void SetCacheSize(int nCacheSize){m_nCacheSize = nCacheSize;}
	virtual void PrintCache() = 0;
	virtual void PrintCachingStatistics()
	{
		int accesses = m_nNumofHits + m_nCompulsoryMisses + m_nCapacityMisses;
		cout << accesses << " accesses, "
			 << m_nNumofHits << " hits, "
			 << m_nCompulsoryMisses << " compulsory misses, "
			 << m_nCapacityMisses << " capacity misses"
			 << endl;
		cout << "cache size v.s. ins is " << m_nCacheSize << " v.s. " << m_nNumofSamples << endl;
/*		ofstream writeOut("cache.txt", ios::app);
		writeOut << GetStrategy() << "hits:\t" << m_nNumofHits
				 << "\t accesses:\t" << accesses << endl;
*/		return;
	}

	//function for cache initialization
	virtual void InitializeCache(const int &nCacheSize, const int &nNumofSamples) = 0;
	virtual string GetStrategy() = 0;
	virtual void CleanCache()
	{
		m_nNumofSamples = 0;
		m_nCacheSize = 0;
		m_nCacheOccupancy = 0;
		m_nNumofHits = 0;
		m_nCompulsoryMisses = 0;
		m_nCapacityMisses = 0;
	}
	//for getting data from cache if data is in cache; otherwise, cache miss.
	virtual bool GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bGetData) = 0;
	//lock an entry in cache
	virtual void LockCacheEntry(int)
	{
		return;
	}
	//unlock an entry in cache
	virtual void UnlockCacheEntry(int)
	{
		return;
	}
	//replace an expired sample
	virtual void ReplaceExpired(int nIndex, int &nLocationInCache, real *pExtraInfo) = 0;
	//check if extra information is required
	virtual bool isExtraInfoRequired()
	{
		return false;
	}
	//update group info
	virtual bool UpdateGroup(real fAlpha, int nLabel, real fCost, int nIndexofSample) = 0;
};

/*
 * @brief: cache using Least Recent Used strategy
 */
class CLRUCache: public virtual CCache
{
public:
	CLRUCache(const int &nNumofSamples):CCache()
	{
		m_nNumofSamples = nNumofSamples;
	}
	virtual~CLRUCache(){}

	virtual void InitializeCache(const int &nCacheSize, const int &nNumofSamples);
	virtual void CleanCache();
	virtual bool GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsIn);
	virtual void ReplaceExpired(int nIndex, int &nLocationInCache, real *pExtraInfo);
	virtual bool isExtraInfoRequired()
	{
		return CCache::isExtraInfoRequired();
	}

	virtual void PrintCache();
	virtual string GetStrategy(){return "LRU";}
	virtual void PrintCachingStatistics();
	virtual bool UpdateGroup(real fAlpha1, int nLabel1, real fCost1, int nIndexofSample1)
	{//LRU doesn't need to update group info, as it doesn't use the it
		return true;
	}

	void InitVector(const int &nNumofSamples);
protected:
	class LRUEntry
	{
	public:
		LRUEntry()
		{
			m_nStatus = NEVER;
			m_nLocationInCache = -1;
		}
		int m_nStatus;
		int m_nLocationInCache;
		list<int>::iterator itLRUList;
	};

	vector<LRUEntry> v_LRUContainer;
	list<int> LRUList;
};

/*
 * @brief: Most Recent Used cache strategy
 */
class CMRUCacheHelper
{
private:
	int m_nReplaceCandidate1;
	int m_nReplaceCandidate2;

public:
	CMRUCacheHelper(){}
	~CMRUCacheHelper(){}

	//candidate1 is the latest used sample
	int GetLatestUsedCandidate();

	//set latest recent used sample
	void SetLatestUsedCandidate(int nIndex);
	//set an expired candidate
	void SetCandidate(int nCandidateIndex);
};

/*
 * @brief: cache first using Most Recent Used and then using Least Recent Used strategy
 */
class CMLRUCache: public CLRUCache
{
//public: CMRUCacheHelper MRUHelper;
public:
	CMLRUCache(const int &nNumofSamples):CLRUCache(nNumofSamples)
	{
		m_nLockedSample = -1;
	}
	~CMLRUCache(){}

	//virtual bool GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsIn);
	virtual string GetStrategy(){return "MLRU";}
	virtual void ReplaceExpired(int nIndex, int &nLocationInCache, real *pExtraInfo);
	int GetExpiredSample();
	virtual void LockCacheEntry(int);
	virtual void UnlockCacheEntry(int);
private:
	int m_nLockedSample;
};

extern int nGlobal;
extern int nInterater;

/*
 * @brief: cache using gradient value based strategy
 */
extern long lCountLatest;
extern long lCountNormal;
class CLATCache: public CCache
{
public:
	CLATCache(const int&);
	~CLATCache(){m_GSContainer.clear();}

	virtual void InitializeCache(const int &nCacheSize, const int &nNumofSamples);
	virtual void CleanCache();
	virtual bool GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsIn);
	virtual void ReplaceExpired(int nIndex, int &nLocationInCache, real *pExtraInfo) ;
	virtual bool isExtraInfoRequired()
	{
		return false;
	}

	virtual void LockCacheEntry(int);
	virtual void UnlockCacheEntry(int);

	virtual string GetStrategy(){return "LAT";}
	virtual void PrintCache();
	virtual bool UpdateGroup(real fAlpha1, int nLabel1, real fCost1, int nIndexofSample1/*,
							 float_point fAlpha2, int nLabel2, float_point fCost2, int nIndexofSample2*/);

	void InitMap(const int &nNumofSamples);
	int GetExpiredSample(real *pGradient, int nIndex);
	int GetExpiredSample2(real *pGradient, int nIndex);
	int GetExpiredSample3(real *pGradient, int nIndex);

	//candidate1 is the latest used sample
	int GetLatestUsedCandidate()
	{
		assert(m_nReplaceCandidate1 != -1 || m_nReplaceCandidate2 != -1);
		if(m_bGetOne)
			return m_nReplaceCandidate1;
		else
			return m_nReplaceCandidate2;

/*		if(m_nReplaceCandidate1 != m_nLockedSample && m_nReplaceCandidate1 != -1)
		{
			return m_nReplaceCandidate1;
		}
		else if(m_nReplaceCandidate2 != m_nLockedSample && m_nReplaceCandidate2 != -1)
		{
			return m_nReplaceCandidate2;
		}
		else
		{
			cerr << "not available sample" << endl;
			return -1;
		}*/
	}

	void SetLatestUsedCandidate(int nIndex)
	{
		if(m_bSetOne)
		{
			m_nReplaceCandidate1 = nIndex;
			m_bSetOne = false;
			m_bGetOne = false;
		}
		else
		{
			m_nReplaceCandidate2 = nIndex;
			m_bSetOne = true;
			m_bGetOne = true;
		}
		/*if(m_nReplaceCandidate1 != -1)
			m_nReplaceCandidate2 = m_nReplaceCandidate1;
		m_nReplaceCandidate1 = nIndex;*/
	}

	void SetCandidate(int nCandidateIndex)
	{
		if(m_nReplaceCandidate1 == nCandidateIndex)
			m_nReplaceCandidate1 = -1;
		else
			m_nReplaceCandidate2 = -1;
	}

private:
	class GSEntry
	{
	public:
		GSEntry()
		{
			m_nStatus = NEVER;
			m_nLocationInCache = -1;
			m_bIsUpSample = true;
			m_bIsLowSample = true;
		}
		int m_nStatus;
		int m_nLocationInCache;

		bool m_bIsUpSample;		//the sample is in I_up set
		bool m_bIsLowSample;	//the sample is in I_low set
		vector<int> m_used;
	};

	vector<GSEntry> m_GSContainer;
	set<int> m_setGS;	//serves as a cache look up table

	int m_nReplaceCandidate1;
	int m_nReplaceCandidate2;
	bool m_bSetOne;
	bool m_bGetOne;
	int m_nLockedSample;

	int m_nMissCount;
	list<int> m_listReplace;
	int m_nTop3rdStableCount;
	int m_nIndexOfTop3;
};

#endif /* CACHE_H_ */
