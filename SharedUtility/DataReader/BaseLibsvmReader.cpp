/*
 * BaseLibsvmReader.cpp
 *
 *  Created on: 6 May 2016
 *      Author: Zeyi Wen
 *		@brief: definition of some basic functions for reading data in libsvm format
 */

#include <iostream>
#include <assert.h>
#include <sstream>
#include <stdio.h>
#include "BaseLibsvmReader.h"

using std::istringstream;

/**
 * @brief: get the number of features and the number of instances of a dataset
 */
void BaseLibSVMReader::GetDataInfo(string strFileName, int &nNumofFeatures, int &nNumofInstance, uint &nNumofValue)
{
	nNumofInstance = 0;
	nNumofFeatures = 0;
	nNumofValue = 0;

	ifstream readIn;
	readIn.open(strFileName.c_str());
	if(readIn.is_open() == false){
		printf("opening %s failed\n", strFileName.c_str());
	}
	assert(readIn.is_open());

	//for storing character from file
	string str;

	//get a sample
	char cColon;
	while (readIn.eof() != true){
		getline(readIn, str);
		if (str == "") break;
		istringstream in(str);

		real fValue = 0;//label
		in >> fValue;

		//get features of a sample
		int nFeature;
		real x = 0xffffffff;
		while (in >> nFeature >> cColon >> x)
		{
			assert(cColon == ':');
			if(nFeature > nNumofFeatures)
				nNumofFeatures = nFeature;
			nNumofValue++;
		}

		//skip an empty line (usually this case happens in the last line)
		if(x == 0xffffffff)
			continue;

		nNumofInstance++;
	};

	//clean eof bit, when pointer reaches end of file
	if(readIn.eof())
	{
		//cout << "end of file" << endl;
		readIn.clear();
	}

	readIn.close();
	 printf("GetDataInfo. # of instances: %d; # of features: %d; # of fvalue: %d\n", nNumofInstance, nNumofFeatures, nNumofValue);
}
