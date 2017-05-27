/**
 * cacheLRU.cpp
 * Created on: May 28, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 * @brief: this file has implementation of the LRU cache strategy
 **/

#include <stdlib.h>
#include "cache.h"

CCache::~CCache()
{

}
/* *
 * @brief: initialize caching vector
 */
void CLRUCache::InitVector(const int &nNumofSamples)
{

	for(int i = 0; i < nNumofSamples; i++)
	{
		LRUEntry lruEntry;
		lruEntry.m_nLocationInCache = -1;
		lruEntry.m_nStatus = NEVER;
		v_LRUContainer.push_back(lruEntry);
	}
	return;
}

/*
 * @brief: initialize cache
 */
void CLRUCache::InitializeCache(const int &nCacheSize, const int &nNumofSamples)
{
	m_nNumofSamples = nNumofSamples;
	v_LRUContainer.reserve(nNumofSamples);
	InitVector(nNumofSamples);
//	cout << "size of cache container " << v_LRUContainer.size() << endl;
}

/*
 * @brief: clean cache
 */
void CLRUCache::CleanCache()
{
	v_LRUContainer.clear();
	LRUList.clear();
	m_nNumofSamples = 0;
	m_nCacheSize = 0;
	m_nCacheOccupancy = 0;
	m_nNumofHits = 0;
	m_nCompulsoryMisses = 0;
	m_nCapacityMisses = 0;
}

/*
 * @brief: get data from cache
 * @nIndex: the index of the target data
 * @nLocationInCache: the location of data entry in cache
 * @bIsCacheFull: variable for checking whether the cache is full
 * @return: true if cache hits, otherwise return false
 */
int nGlobal = 0;
bool CLRUCache::GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsCacheFull)
{
	nGlobal++;
	assert(nIndex > v_LRUContainer.size() && nIndex >= 0 && bIsCacheFull == false);
	vector<LRUEntry>::iterator itCheckEntry = v_LRUContainer.begin() + nIndex; //check the container
	if(itCheckEntry->m_nStatus == CACHED)
	{//in cache
		m_nNumofHits++;
		if(itCheckEntry->itLRUList != LRUList.begin())
		{//put the latest used entry to the front of the list
			LRUList.erase(itCheckEntry->itLRUList);
			LRUList.push_front(nIndex);
			itCheckEntry->itLRUList = LRUList.begin();
		}

		nLocationInCache = itCheckEntry->m_nLocationInCache;
		bIsCacheFull = false;
		return true;//cache hits
	}

	//Cache Miss
	if(m_nCacheOccupancy < m_nCacheSize)
	{
		//Cache has empty space
		m_nCompulsoryMisses++;
		itCheckEntry->m_nLocationInCache = m_nCacheOccupancy;
		nLocationInCache = m_nCacheOccupancy;
		itCheckEntry->m_nStatus = CACHED;
		LRUList.push_front(nIndex);
		itCheckEntry->itLRUList = LRUList.begin();
		m_nCacheOccupancy++;
		bIsCacheFull = false;
//		cout << "cache list " << LRUList.size() << endl;
		return false;//cache misses
	}

	assert(nLocationInCache < m_nCacheSize);
	//Cache is full
//	cout << "cache is full; occupancy is " << m_nCacheOccupancy << "; cache size is " << m_nCacheSize << endl;
	bIsCacheFull = true;
	return false;
}

/*
 * @brief: replace expired sample
 */
void CLRUCache::ReplaceExpired(int nIndex, int &nLocationInCache, real *pExtraInfo)
{
	vector<LRUEntry>::iterator itCheckEntry = v_LRUContainer.begin() + nIndex;
	if(itCheckEntry->m_nStatus == NEVER)
	{
		m_nCompulsoryMisses++;
	}
	else
	{
		m_nCapacityMisses++;
	}

	if(LRUList.empty())
	{
		cerr << "error at GetDataFromCache" << endl;
		return;
	}

	//replace an sample
	int nExpiredSample = LRUList.back();

	int nTempLocationInCache = v_LRUContainer[nExpiredSample].m_nLocationInCache;
	v_LRUContainer[nExpiredSample].itLRUList = LRUList.end();
	v_LRUContainer[nExpiredSample].m_nLocationInCache = -1;

	LRUList.pop_back();

	v_LRUContainer[nExpiredSample].m_nStatus = EVICTED;

	itCheckEntry->m_nStatus = CACHED;
	itCheckEntry->m_nLocationInCache = nTempLocationInCache;//v_LRUContainer[nExpiredSample].m_nLocationInCache; //Zeyi: location in the cache
	LRUList.push_front(nIndex);
	itCheckEntry->itLRUList = LRUList.begin();

	nLocationInCache = itCheckEntry->m_nLocationInCache;

	return;
}

void CLRUCache::PrintCachingStatistics()
{
	int accesses = m_nNumofHits + m_nCompulsoryMisses + m_nCapacityMisses;
	cout << accesses << " accesses, "
		 << m_nNumofHits << " hits, "
		 << m_nCompulsoryMisses << " compulsory misses, "
		 << m_nCapacityMisses << " capacity misses"
		 << endl;
	cout << "cache size v.s. ins is " << m_nCacheSize << " v.s. " << m_nNumofSamples << endl;
	cout << "cache occupancy is " << m_nCacheOccupancy << endl;
	return;
}

void CLRUCache::PrintCache()
{
  int accesses = m_nNumofHits + m_nCompulsoryMisses + m_nCapacityMisses;
  float hitPercent = (float)m_nNumofHits*100.0/float(accesses);
  float compulsoryPercent = (float)m_nCompulsoryMisses*100.0/float(accesses);
  float capacityPercent = (float)m_nCapacityMisses*100.0/float(accesses);

  cout << "Cache hits: " << hitPercent
	   << " compulsory misses: " << compulsoryPercent
	   << " capacity misses" << capacityPercent
	   << endl;
  //printf("Cache hits: %f compulsory misses: %f capacity misses %f\n", hitPercent, compulsoryPercent, capacityPercent);
  for(int i = 0; i < m_nNumofSamples; i++)
  {
    if (v_LRUContainer[i].m_nStatus == CACHED)
    {
    	cout << "Row " << i << ": present @ cache line " << v_LRUContainer[i].m_nLocationInCache << endl;
      //printf("Row %d: present @ cache line %d\n", i, v_LRUContainer[i].m_nLocationInCache);
    }
    else
    {
    	cout << "Row " << i << ": not present" << endl;
      //printf("Row %d: not present\n", i);
    }
  }
  cout << "----" << endl;
  list<int>::iterator i = LRUList.begin();
  for(;i != LRUList.end(); i++)
  {
    cout << "Offset: " <<  *i << endl;;
  }
}

