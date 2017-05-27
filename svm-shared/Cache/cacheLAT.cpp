/**
 * cacheGS.cpp
 * Created on: April 13, 2013
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 * @brief: this file has implementation of the gradient value based cache strategy
 **/
#include "../my_assert.h"
#include <stdlib.h>
#include <float.h>
#include "cache.h"

using std::make_pair;

/**
 * @brief: CGradientStrategy constructor
 */
CLATCache::CLATCache(const int &nNumofSamples):CCache()
{
	m_nNumofSamples = nNumofSamples;
	m_nReplaceCandidate1 = -1;
	m_nReplaceCandidate2 = -1;
	m_nLockedSample = -1;
	m_bSetOne = true;
	m_bGetOne = true;
	m_listReplace.clear();
	m_nTop3rdStableCount = 0;
	m_nIndexOfTop3 = 0;
}

/* *
 * @brief: initialise caching map
 */
void CLATCache::InitMap(const int &nNumofSamples)
{
	GSEntry hlruEntry;
	for(int i = 0; i < nNumofSamples; i++)
	{
		m_GSContainer.push_back(hlruEntry);
	}
	return;
}

/*
 * @brief: initialise cache
 */
void CLATCache::InitializeCache(const int &nCacheSize, const int &nNumofSamples)
{
	m_nNumofSamples = nNumofSamples;
	InitMap(nNumofSamples);
}

/*
 * @brief: clean cache
 */
void CLATCache::CleanCache()
{
	m_GSContainer.clear();
	m_setGS.clear();
	CCache::CleanCache();
}

/*
 * @brief: get data from cache
 * @nIndex: the index of the target data
 * @nLocationInCache: the location of data entry in cache
 * @bIsCacheFull: variable for checking whether the cache is full
 * @return: true if cache hits, otherwise return false
 */
bool CLATCache::GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsCacheFull)
{
	nGlobal++;
	assert(nIndex < m_GSContainer.size() && nIndex >= 0 && m_GSContainer.size() == m_nNumofSamples);

	vector<GSEntry>::iterator itCheckEntry = m_GSContainer.begin() + nIndex; //check if the entry is cached in the container
	assert(itCheckEntry != m_GSContainer.end());
	assert(itCheckEntry->m_nStatus == CACHED || itCheckEntry->m_nStatus == NEVER || itCheckEntry->m_nStatus == EVICTED);

//	itCheckEntry->m_used.push_back(nInterater);
	//PrintCacheContent(nIndex);

	if(itCheckEntry->m_nStatus == CACHED)
	{//in cache
		m_nMissCount = 0;
		m_nNumofHits++;
		nLocationInCache = itCheckEntry->m_nLocationInCache;
		assert(nLocationInCache >= 0);	//check if the cache location is valid
		bIsCacheFull = false;
		//set latest used
		SetLatestUsedCandidate(nIndex);
		return true;
	}
	//Cache Miss
	else if(m_nCacheOccupancy < m_nCacheSize)
	{
		m_nMissCount = 0;
		m_nCompulsoryMisses++;
		itCheckEntry->m_nLocationInCache = m_nCacheOccupancy;
		itCheckEntry->m_nStatus = CACHED;

		m_nCacheOccupancy++;
		nLocationInCache = itCheckEntry->m_nLocationInCache;

		//update look up set
		//assert(m_setGS.find(nIndex) == m_setGS.end());
		m_setGS.insert(nIndex);

		bIsCacheFull = false;
		//set latest used
		SetLatestUsedCandidate(nIndex);
		return false;
	}
	else
	{
		//Cache is full
		bIsCacheFull = true;
		return false;
	}
}

/*
 * @brief: replace expired sample
 */
void CLATCache::ReplaceExpired(int nIndex, int &nLocationInCache, real *pGradient)
{
	m_nMissCount++;
	vector<GSEntry>::iterator itCheckEntry = m_GSContainer.begin() + nIndex; //check if the entry is cached in the container
	if (itCheckEntry->m_nStatus == NEVER)
	{
		m_nCompulsoryMisses++;
	}
	else
	{
		m_nCapacityMisses++;
	}

	//replace an sample
	int nExpiredSample = GetExpiredSample(pGradient, nIndex);
	assert(nExpiredSample != -1);

	//assert(m_setGS.find(nIndex) == m_setGS.end());
	//assert(m_setGS.find(nExpiredSample) != m_setGS.end());
	//erase the expired sample
	m_setGS.erase(nExpiredSample);
	//insert the new sample into the look up set
	m_setGS.insert(nIndex);

	//update cache status for the expired sample
	int nTempLocationInCache = m_GSContainer[nExpiredSample].m_nLocationInCache;
	assert(nTempLocationInCache != -1);
	m_GSContainer[nExpiredSample].m_nLocationInCache = -1;
	m_GSContainer[nExpiredSample].m_nStatus = EVICTED;

	//update cache status for the new entered sample
	itCheckEntry->m_nStatus = CACHED;
	itCheckEntry->m_nLocationInCache = nTempLocationInCache; //v_LRUContainer[nExpiredSample].m_nLocationInCache; //Zeyi: location in the cache

	nLocationInCache = itCheckEntry->m_nLocationInCache;
	//set latest used
	SetLatestUsedCandidate(nIndex);
}

/*
 * @brief: get expired sample from cache based on gradient value
 * @param: pGradient is the gradient values of training samples
 * @param: nIndex is the index of sample which enters cache
 */
int CLATCache::GetExpiredSample(real *pGradient, int nIndex)
{
	//remove the row with the smallest id
	return *m_setGS.begin();
}

/*
 * @brief: update group
 */
bool CLATCache::UpdateGroup(real fAlpha, int nLabel, real fCost, int nIndexofSample)
{
	if((fAlpha < fCost && nLabel > 0) || (fAlpha > 0 && nLabel < 0))
	{
		if(m_GSContainer[nIndexofSample].m_bIsUpSample == false)
		{
			//cerr << "very good low --> up" << endl;
		}
		m_GSContainer[nIndexofSample].m_bIsUpSample = true;
	}
	else
	{
		//cout << "remove from up" << endl;
		m_GSContainer[nIndexofSample].m_bIsUpSample = false;
	}

	if((fAlpha > 0 && nLabel > 0) || (fAlpha < fCost && nLabel < 0))
	{
		if(m_GSContainer[nIndexofSample].m_bIsLowSample == false)
		{
			//cerr << "very good up --> low" << endl;
		}
		m_GSContainer[nIndexofSample].m_bIsLowSample = true;
	}
	else
	{
		m_GSContainer[nIndexofSample].m_bIsLowSample = false;
		//cout << "remove from low" << endl;
	}

	if(m_GSContainer[nIndexofSample].m_bIsLowSample == false && m_GSContainer[nIndexofSample].m_bIsUpSample == false)
	{
		cerr << "error " << endl;
	}

	return true;
}

/*
 * @brief: lock an cached entry
 * @param: nIndex: entry index in gradient array
 */
void CLATCache::LockCacheEntry(int nIndex)
{
	m_nLockedSample = nIndex;
	assert(m_setGS.find(nIndex) != m_setGS.end());
	m_setGS.erase(nIndex);
}

/*
 * @brief: unlock an cached entry
 * @param: nIndex: entry index in gradient array
 */
void CLATCache::UnlockCacheEntry(int nIndex)
{
	m_nLockedSample = -1;
	assert(m_setGS.find(nIndex) == m_setGS.end());
	m_setGS.insert(nIndex);
}

/*
 * @brief: check the content of the cache
 */
void CLATCache::PrintCache()
{
	vector<GSEntry>::iterator itCheckEntry = m_GSContainer.begin(); //check if the entry is cached in the container
	ofstream out;
	out.open("cache_stat2.txt", ios::out | ios::app);
	for(int i = 0; i < int(m_GSContainer.size()); i++)
	{
		out << i << "\t" << itCheckEntry->m_used.size();
		/*for(int j = 0; j < itCheckEntry->m_used.size(); j++)
		{
			out << itCheckEntry->m_used[j] << " ";
		}*/
		out << endl;
		itCheckEntry++;
	}

	out << "######################################" << endl;
}
