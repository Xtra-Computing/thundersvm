/*
 * ReadAMZ.cpp
 *
 *  Created on: 21/11/2013
 *      Author: zeyi
 */

#include "DataIO.h"
#include <assert.h>

bool CReadAMZ::ReadFromFile(string strFileName, vector<vector<float_point> > &v_vSampleData, vector<float> &v_fValue,
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
