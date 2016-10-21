/*
 * accessHessian.cpp
 *
 *  Created on: 30/10/2015
 *      Author: Zeyi Wen
 */

#include <assert.h>
#include "accessHessian.h"

int HessianAccessor::m_nTotalNumofInstance = -1;

//for Hessian operation in n-fold-cross-validation
int HessianAccessor::m_nRowStartPos1 = -1;	//index of the fisrt part of samples
int HessianAccessor::m_nRowEndPos1 = -1;		//index of the end of the first part of samples (include the last sample)
int HessianAccessor::m_nRowStartPos2 = -1;	//index of the second part of samplels
int HessianAccessor::m_nRowEndPos2 = -1;		//index of the end of the second part of samples (include the last sample)

/*
 * @brief: set data involved in Hessian Read Operation
 * @param: nStart1: the index of the first part of a row
 * @param: nEnd1: the index of the end of the first part of a row
 * @param: nStart2: the index of the second part of a row
 * @param: nEnd2: the index of the end of the second part of a row
 */
void HessianAccessor::SetInvolveData(const int &nStart1, const int &nEnd1, const int &nStart2, const int &nEnd2)
{
	assert(nStart1 < m_nTotalNumofInstance && nEnd1 < m_nTotalNumofInstance &&
		   nStart2 < m_nTotalNumofInstance && nEnd2 < m_nTotalNumofInstance);
	m_nRowStartPos1 = nStart1;
	m_nRowEndPos1 = nEnd1;
	m_nRowStartPos2 = nStart2;
	m_nRowEndPos2 = nEnd2;
}
