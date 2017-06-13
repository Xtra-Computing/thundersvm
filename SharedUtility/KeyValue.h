/*
 * keyValue.h
 *
 *  Created on: 9 Mar 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef KEYVALUE_H_
#define KEYVALUE_H_

#include <vector>
#include "DataType.h"


using std::vector;

class KeyValue
{
public:
	real featureValue;
	int id;//This id may be used as feature id or instance id

public:
	static void SortFeaValue(int nNumofDim, vector<vector<KeyValue> > &vvInsSparse, vector<vector<KeyValue> > &vvFeaInxPair);
	static void VecToArray(vector<vector<KeyValue> > &vvFeaInxPair, int *pInsId, real *pdValue, int *pNumofKeyValue, long long *plStartPos);
};



#endif /* KEYVALUE_H_ */
