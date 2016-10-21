/*
 * BaseLibsvmReader.h
 *
 *  Created on: 6 May 2016
 *      Author: Zeyi Wen
 *		@brief: a class that contains some basic functions for reading data in libsvm format
 */

#ifndef BASELIBSVMREADER_H_
#define BASELIBSVMREADER_H_

#include <fstream>

using std::string;
using std::ifstream;

typedef float float_point;

class BaseLibSVMReader
{
public:
	static void GetDataInfo(string strFileName, int &nNumofFeatures, int &nNumofInstance, long long &nNumofValue);
};



#endif /* BASELIBSVMREADER_H_ */
