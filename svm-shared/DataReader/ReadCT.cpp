/*
 * ReadCT.cpp
 *
 *  Created on: 14/11/2013
 *      Author: zeyi
 */

#include "DataIO.h"
#include <iostream>
#include <assert.h>
#include <sstream>

using std::istringstream;
using std::cout;
using std::endl;

bool CReadCT::ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<float> &v_fValue,
							  int nNumofInstance, int nDim)
{
	bool nReturn = true;
	//these two containers is for storing positive and negative samples from file respectively
	//read data from file
	ifstream readIn;
	readIn.open(strFileName.c_str());
	assert(readIn.is_open());
	vector<real> vSample;

	//for storing character from file
	int j = 0;
	string str;
	getline(readIn, str);//ignore the first line

	//get a sample
	char cColon;
	do {
		j++;
		getline(readIn, str);

		istringstream in(str);
		int i = 0;
		float Id = 0;
		in >> Id >> cColon;//ignore

		//get features of a sample
		real x;
		while (in >> x)
		{
			i++;
			if(i == 385)
				break;
			if(i == 241)
			{
				v_fValue.push_back(x);
				in >> cColon;
				continue;
			}
			vSample.push_back(x);
			in >> cColon;
		}
		v_vSampleData.push_back(vSample);

		//clear vector
		vSample.clear();
	} while (readIn.eof() != true && j < nNumofInstance);///72309 is the number of samples

	//clean eof bit, when pointer reaches end of file
	if(readIn.eof())
	{
		//cout << "end of file" << endl;
		readIn.clear();
	}

	//organize the samples
	assert(int(v_vSampleData.size()) == nNumofInstance);

	return nReturn;
}
