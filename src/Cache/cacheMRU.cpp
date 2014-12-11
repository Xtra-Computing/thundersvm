/*
 * cacheMRU.cpp
 * Created on: 25/04/2013
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 * @brief: this file has implementation of the Most Recent Used cache strategy
 */

#include "cache.h"

/*
 * @brief: get the latest used sample
 */
int CMRUCacheHelper::GetLatestUsedCandidate()
{
	assert(m_nReplaceCandidate1 != -1 || m_nReplaceCandidate2 != -1);
	if(m_nReplaceCandidate1 != -1)
	{
		return m_nReplaceCandidate1;
	}
	else if(m_nReplaceCandidate2 != -1)
	{
		return m_nReplaceCandidate2;
	}
	else
	{
		cerr << "not available sample" << endl;
		return -1;
	}
}

//set latest recent used sample
void CMRUCacheHelper::SetLatestUsedCandidate(int nIndex)
{
	if(m_nReplaceCandidate1 != -1)
		m_nReplaceCandidate2 = m_nReplaceCandidate1;
	m_nReplaceCandidate1 = nIndex;
}

//set an expired candidate
void CMRUCacheHelper::SetCandidate(int nCandidateIndex)
{
	if(m_nReplaceCandidate1 == nCandidateIndex)
		m_nReplaceCandidate1 = -1;
	else
		m_nReplaceCandidate2 = -1;
}

