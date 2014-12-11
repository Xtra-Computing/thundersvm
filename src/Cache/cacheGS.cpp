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
CGradientStrategy::CGradientStrategy(const int &nNumofSamples):CCache()
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
 /*
 * @brief: initialise caching map
 */
void CGradientStrategy::InitMap(const int &nNumofSamples)
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
void CGradientStrategy::InitializeCache(const int &nCacheSize, const int &nNumofSamples)
{
	m_nNumofSamples = nNumofSamples;
	InitMap(nNumofSamples);
}

/*
 * @brief: clean cache
 */
void CGradientStrategy::CleanCache()
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
bool CGradientStrategy::GetDataFromCache(const int &nIndex, int &nLocationInCache, bool &bIsCacheFull)
{
	nGlobal++;
	assert(nIndex < m_GSContainer.size() && nIndex >= 0 && m_GSContainer.size() == m_nNumofSamples);

	vector<GSEntry>::iterator itCheckEntry = m_GSContainer.begin() + nIndex; //check if the entry is cached in the container
	assert(itCheckEntry != m_GSContainer.end());
	assert(itCheckEntry->m_nStatus == CACHED || itCheckEntry->m_nStatus == NEVER || itCheckEntry->m_nStatus == EVICTED);

	itCheckEntry->m_used.push_back(nInterater);
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
void CGradientStrategy::ReplaceExpired(int nIndex, int &nLocationInCache, float_point *pGradient)
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
int nCount = 0;
long lCountLatest = 0;
long lCountNormal = 0;
int CGradientStrategy::GetExpiredSample2(float_point *pGradient, int nIndex)
{
	//return *(m_setGS.rbegin());
	int nExpiredSample = -1;
	//cache replacement indicators
	float_point fIupMax = -FLT_MAX;
	float_point fIupMin = FLT_MAX;
	float_point fIlowMax = -FLT_MAX;
	float_point fIlowMin = FLT_MAX;

	int nIndexofIupMax = -1, nIndexofIupMin = -1, nIndexofIlowMax = -1, nIndexofIlowMin = -1;

	int nSizeofCache = m_setGS.size();
	assert(nSizeofCache == m_nCacheSize || nSizeofCache == m_nCacheSize - 1);//in some cases, one of the entries is locked.

	//get I_up min/max and I_low min/max
	set<int>::iterator itSet = m_setGS.begin();
	int nUpCounter = 0;
	int nLowCounter = 0;
	for(int i = 0; i < nSizeofCache; i++)
	{
		if(*itSet == m_nLockedSample)
			continue;//skip the locked sample

		bool bIsIup = m_GSContainer[*itSet].m_bIsUpSample;
		bool bIsIlow = m_GSContainer[*itSet].m_bIsLowSample;

		if(bIsIup)
		{
			nUpCounter++;
			if(pGradient[*itSet] > fIupMax)
			{
				fIupMax = pGradient[*itSet];
				nIndexofIupMax = *itSet;
			}
			if(pGradient[*itSet] < fIupMin)
			{
				fIupMin = pGradient[*itSet];
				nIndexofIupMin = *itSet;
			}
		}
		if(bIsIlow)
		{
			nLowCounter++;
			if(pGradient[*itSet] > fIlowMax)
			{
				fIlowMax = pGradient[*itSet];
				nIndexofIlowMax = *itSet;
			}
			if(pGradient[*itSet] < fIlowMin)
			{
				fIlowMin = pGradient[*itSet];
				nIndexofIlowMin = *itSet;
			}
		}
		itSet++;
	}

	assert(nIndexofIlowMin != -1 && nIndexofIlowMax != -1 && nIndexofIupMin != -1 && nIndexofIupMax != -1);

	//check which group the new entered sample belongs to
	itSet = m_setGS.begin();
	bool bIsInIup = m_GSContainer[nIndex].m_bIsUpSample;
	bool bIsInIlow = m_GSContainer[nIndex].m_bIsLowSample;

	lCountNormal++;
	if(bIsInIup)
	{
		//if the the max values of up and low conflict
		if(nIndexofIupMax == nIndexofIlowMax)
		{
			//if the min values of up and low conflict
			if(nIndexofIlowMin == nIndexofIupMin)
			{
				lCountLatest++;
				nExpiredSample = GetLatestUsedCandidate();
				SetCandidate(nExpiredSample);
			}
			else
			{
				nExpiredSample = nIndexofIlowMin;
			}
		}
		else
		{
			//cout << "good up" << endl;
			nExpiredSample = nIndexofIupMax;
		}
	}
	else if(bIsInIlow)
	{
		////if the min values of up and low conflict
		if(nIndexofIlowMin == nIndexofIupMin)
		{
			//if the the max values of up and low conflict
			if(nIndexofIupMax == nIndexofIlowMax)
			{
				lCountLatest++;
				nExpiredSample = GetLatestUsedCandidate();
				SetCandidate(nExpiredSample);
			}
			else
			{
				nExpiredSample = nIndexofIupMax;
			}
		}
		else
		{
			nExpiredSample = nIndexofIlowMin;
		}
	}
	else
	{
		cerr << "error in GetExpiredSample" << endl;
		exit(1);
	}

	return nExpiredSample;
}

int CGradientStrategy::GetExpiredSample(float_point *pGradient, int nIndex)
{
	/*if(m_nCapacityMisses == 120995 || m_nMissCount > (m_nNumofSamples / m_nCacheSize))
	{
		m_nMissCount = 0;
		cout << m_nCapacityMisses << endl;
		set<int>::iterator it = m_setGS.begin();
		for(int i = 0; i < m_setGS.size(); i++)
		{
			cout << *it << " ";
			if(i % 50 == 0)cout << endl;
			it++;
		}

	}
	if((*m_setGS.begin()) == (m_nNumofSamples - m_nCacheSize - 1))
		exit(0);*/
	return *m_setGS.begin();
	int nExpiredSample = -1;
	//cache replacement indicators
	float_point fIupMax = pGradient[m_nReplaceCandidate1];
	float_point fIlowMin = pGradient[m_nReplaceCandidate2];

	int nIndexofIupMax = m_nReplaceCandidate1, nIndexofIlowMin = m_nReplaceCandidate2;

	int nSizeofCache = m_setGS.size();
	assert(nSizeofCache == m_nCacheSize || nSizeofCache == m_nCacheSize - 1);//in some cases, one of the entries is locked.

	//get I_up min/max and I_low min/max
	set<int>::iterator itSet = m_setGS.begin();
	for(int i = 0; i < nSizeofCache; i++)
	{
		bool bIsIup = m_GSContainer[*itSet].m_bIsUpSample;
		bool bIsIlow = m_GSContainer[*itSet].m_bIsLowSample;

		if(bIsIup && m_bGetOne && !bIsIlow)
		{
			if(pGradient[*itSet] > fIupMax)
			{
				//cout << "hi" << endl;
				fIupMax = pGradient[*itSet];
				nIndexofIupMax = *itSet;
			}
		}
		if(bIsIlow && !m_bGetOne && !bIsIup)
		{
			//cout << "hi" << endl;
			if(pGradient[*itSet] < fIlowMin)
			{
				fIlowMin = pGradient[*itSet];
				nIndexofIlowMin = *itSet;
			}
		}
		itSet++;
	}

	if(m_bGetOne)
		nExpiredSample = nIndexofIupMax;
	else
		nExpiredSample = nIndexofIlowMin;


	return nExpiredSample;
}

int CGradientStrategy::GetExpiredSample3(float_point *pGradient, int nIndex)
{
	set<int>::reverse_iterator itSet = m_setGS.rbegin();
	if(*itSet == m_nIndexOfTop3)
		m_nTop3rdStableCount++;
	else
	{
		m_nIndexOfTop3 = *itSet;
		m_nTop3rdStableCount = 0;
	}
	if(m_nTop3rdStableCount > (m_nNumofSamples / m_nCacheSize))
	{
		m_nMissCount = 0;
		int nExpiredSample = -1;
		//cache replacement indicators
		float_point fIupMax = pGradient[m_nReplaceCandidate1];
		float_point fIlowMin = pGradient[m_nReplaceCandidate2];

		int nSizeofCache = m_setGS.size();
		assert(nSizeofCache == m_nCacheSize || nSizeofCache == m_nCacheSize - 1);//in some cases, one of the entries is locked.

		//get I_up min/max and I_low min/max
		set<int>::reverse_iterator it = m_setGS.rbegin();
		for(int i = 0; i < nSizeofCache; i++)
		{
			bool bIsIup = m_GSContainer[*it].m_bIsUpSample;
			bool bIsIlow = m_GSContainer[*it].m_bIsLowSample;

			if(bIsIup && !bIsIlow)
			{
				if(pGradient[*it] > fIupMax)
				{
					m_listReplace.push_back(*it);
				}
			}
			if(bIsIlow && !bIsIup)
			{
				if(pGradient[*it] < fIlowMin)
				{
					m_listReplace.push_back(*it);
				}
			}
			if(m_listReplace.size() >= 10)break;
			it++;
		}
	}

	if(!m_listReplace.empty())
	{
		m_nIndexOfTop3 = 0;
		m_nTop3rdStableCount = 0;
		m_nMissCount = 0;
		int nReturn = m_listReplace.front();
		/*list<int>::iterator it = m_listReplace.begin();
		for(int i = 0; i < m_listReplace.size(); i++)
		{
			cout << *it << " ";
			it++;
		}
		cout << endl;*/
		m_listReplace.pop_front();
		if(nReturn == m_nLockedSample)
		{
			if(m_listReplace.empty())
			{
				return *m_setGS.begin();
			}
			nReturn = m_listReplace.front();
			m_listReplace.pop_front();
		}

		cout << "hi: " << nReturn << endl;
		return nReturn;
	}
	else
	{
		return *m_setGS.begin();
	}
}

/*
 * @brief: update group
 */
bool CGradientStrategy::UpdateGroup(float_point fAlpha, int nLabel, float_point fCost, int nIndexofSample)
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
void CGradientStrategy::LockCacheEntry(int nIndex)
{
	m_nLockedSample = nIndex;
	assert(m_setGS.find(nIndex) != m_setGS.end());
	m_setGS.erase(nIndex);
}

/*
 * @brief: unlock an cached entry
 * @param: nIndex: entry index in gradient array
 */
void CGradientStrategy::UnlockCacheEntry(int nIndex)
{
	m_nLockedSample = -1;
	assert(m_setGS.find(nIndex) == m_setGS.end());
	m_setGS.insert(nIndex);
}

/*
 * @brief: check the content of the cache
 */
void CGradientStrategy::PrintCache()
{
	vector<GSEntry>::iterator itCheckEntry = m_GSContainer.begin(); //check if the entry is cached in the container
	ofstream out;
	out.open("cache_stat2.txt", ios::out | ios::app);
	for(int i = 0; i < m_GSContainer.size(); i++)
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

	/*exit(0);*/

/*	set<int>::iterator itSetEntry = m_setGS.begin();
	for(int i = 0; i < m_setGS.size(); i++)
	{
		cout << *itSetEntry << ", ";
		itSetEntry++;
	}
	cout << endl;
	sleep(1);*/
}
