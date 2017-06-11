/*
 * cacheMLRU.cpp
 *
 *  Created on: 24/04/2013
 *      Author: Zeyi Wen
 *  Copyright @DBGroup University of Melbourne
 * @brief: this file has implementation of the MLRU cache strategy
 */

#include "cache.h"
#include <stdlib.h>

/*bool CMLRUCache::GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsCacheFull)
{
	nGlobal++;
	assert(nIndex < v_LRUContainer.size() && nIndex >= 0);

	vector<LRUEntry>::iterator itCheckEntry = v_LRUContainer.begin() + nIndex; //check the container
	if(itCheckEntry->m_nStatus == CACHED)
	{
		//MRUHelper.SetLatestUsedCandidate(nIndex);
		m_nNumofHits++;
		if(itCheckEntry->itLRUList == LRUList.begin())
		{
			nLocationInCache = itCheckEntry->m_nLocationInCache; //assign location
			bIsCacheFull = false;
			return true;//cache hits
		}

		LRUList.erase(itCheckEntry->itLRUList);
		LRUList.push_front(nIndex);
		itCheckEntry->itLRUList = LRUList.begin();

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
		itCheckEntry->m_nStatus = CACHED;
		LRUList.push_front(nIndex);
		itCheckEntry->itLRUList = LRUList.begin();
		m_nCacheOccupancy++;
		nLocationInCache = itCheckEntry->m_nLocationInCache;
		bIsCacheFull = false;

		//MRUHelper.SetLatestUsedCandidate(nIndex);
		return false;//cache misses
	}

	//Cache is full
	bIsCacheFull = true;
	return false;
}*/

/*
 * @brief: replace expired sample
 */
void CMLRUCache::ReplaceExpired(int nIndex, int &nLocationInCache, real *pExtraInfo)
{

	vector<LRUEntry>::iterator itCheckEntry = v_LRUContainer.begin() + nIndex;
	assert(itCheckEntry->m_nStatus != CACHED);

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
	int nExpiredSample = GetExpiredSample();
	assert(v_LRUContainer[nExpiredSample].m_nStatus == CACHED);
	assert(nExpiredSample > -1);

	int nTempLocationInCache = v_LRUContainer[nExpiredSample].m_nLocationInCache;
	LRUList.erase(v_LRUContainer[nExpiredSample].itLRUList);//this is for MLRU

	v_LRUContainer[nExpiredSample].itLRUList = LRUList.end();
	v_LRUContainer[nExpiredSample].m_nLocationInCache = -1;
	v_LRUContainer[nExpiredSample].m_nStatus = EVICTED;

	itCheckEntry->m_nStatus = CACHED;
	itCheckEntry->m_nLocationInCache = nTempLocationInCache;//location in the cache
	LRUList.push_front(nIndex);
	itCheckEntry->itLRUList = LRUList.begin();

	nLocationInCache = itCheckEntry->m_nLocationInCache;
	//set latest used
	//MRUHelper.SetLatestUsedCandidate(nIndex);
	return;
}

int CMLRUCache::GetExpiredSample()
{
	int nReturn = - 1;
	//use LRUaccesses
	/*if(m_nCompulsoryMisses + m_nCapacityMisses + m_nNumofHits > 120000)
		nReturn = LRUList.back();
	else*/
	{
		nReturn = LRUList.front();
		if(nReturn == m_nLockedSample)
		{
			list<int>::iterator it = LRUList.begin();
			nReturn = *(++it);
		}
	}
	/*int r = rand() % LRUList.size();
	list<int>::iterator it = LRUList.begin();
	for(int i = 0; i < r; i++)
	{
		it++;
	}
		nReturn = *it;
		if(nReturn == m_nLockedSample)
		{
			//list<int>::iterator it = LRUList.begin();
			if(r < LRUList.size())
				nReturn = *(++it);
			else
				nReturn = *(--it);
		}
*/
	return nReturn;
}

void CMLRUCache::LockCacheEntry(int nIndex)
{
	m_nLockedSample = nIndex;
}

void CMLRUCache::UnlockCacheEntry(int nIndex)
{
	m_nLockedSample = -1;
}
