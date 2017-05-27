/*
 * ReadAdultData.cpp
 *
 *  Created on: 26/04/2013
 *  Author: Zeyi Wen
 */

#include "DataIO.h"
#include <assert.h>

bool CReadE2006::ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<float> &v_fValue,
							  int nNumofInstance, int nDim)
{
	bool nReturn = true;
	//these two containers is for storing positive and negative samples from file respectively
	//read data from file
	ReadLibSVMDataFormat(v_vSampleData, v_fValue, strFileName, nDim, nNumofInstance);
	//organize the samples
	assert(int(v_vSampleData.size()) == nNumofInstance);

	return nReturn;
}

