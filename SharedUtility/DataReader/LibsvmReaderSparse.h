/**
 * trainingDataIO.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 **/

#ifndef TRAININGDATAIO_H_
#define TRAININGDATAIO_H_


#include <assert.h>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <limits>
#include "BaseLibsvmReader.h"
#include "../DataType.h"
#include "../KeyValue.h"

using std::string;
using std::vector;
using std::istringstream;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

class LibSVMDataReader: public BaseLibSVMReader
{
public:
	LibSVMDataReader(){}
	~LibSVMDataReader(){}

    template<class T>
	void ReadLibSVMAsDense(vector<vector<real> > &v_vSample, vector<T> &v_targetValue,
							  string strFileName, int nNumofFeatures, int nNumofInstance = -1);

    template<class T>
	void ReadLibSVMAsSparse(vector<vector<KeyValue> > &v_vSample, vector<T> &v_targetValue,
			  	  	  	  	  	string strFileName, int nNumofFeatures, int nNumofInstance = -1);

private:
    template<class T>
	void ReaderHelper(vector<vector<KeyValue> > &v_vSample, vector<T> &v_targetValue,
	  	  	  		  string strFileName, int nNumofFeatures, int nNumofInstance, bool bUseDense);
	void Push(int feaId, real value, vector<KeyValue> &vIns){
        KeyValue pair;
        pair.id = feaId;
        pair.featureValue = value;
        vIns.push_back(pair);
    }
};

/**
 * @brief: represent the data in a sparse form
 */
template<class T>
void LibSVMDataReader::ReadLibSVMAsSparse(vector<vector<KeyValue> > &v_vInstance, vector<T> &v_targetValue,
											  string strFileName, int nNumofFeatures, int nNumofInstance)
{
	if(nNumofInstance == -1){
		nNumofInstance = std::numeric_limits<int>::max();
	}
    cout << "reading libsvm data as sparse format..." << endl;
	ReaderHelper(v_vInstance, v_targetValue, strFileName, nNumofFeatures, nNumofInstance, false);
}

/**
 * @brief: store the instances in a dense form
 */
template<class T>
void LibSVMDataReader::ReadLibSVMAsDense(vector<vector<real> > &v_vInstance, vector<T> &v_targetValue,
									  	    string strFileName, int nNumofFeatures, int nNumofExamples)
{
	if(nNumofExamples == -1){
		nNumofExamples = std::numeric_limits<int>::max();
	}
	vector<vector<KeyValue> > v_vInstanceKeyValue;
    cout << "reading libsvm data as dense format..." << endl;
	ReaderHelper(v_vInstanceKeyValue, v_targetValue, strFileName, nNumofFeatures, nNumofExamples, true);

	//convert key values to values only.
	for(int i = 0; i < v_vInstanceKeyValue.size(); i++)
	{
		vector<real> vIns;
		for(int j = 0; j < nNumofFeatures; j++)
		{
			vIns.push_back(v_vInstanceKeyValue[i][j].featureValue);
		}
		v_vInstance.push_back(vIns);
	}
}

/**
 * @brief: a function to read instances from libsvm format as either sparse or dense instances.
 */
template<class T>
void LibSVMDataReader::ReaderHelper(vector<vector<KeyValue> > &v_vInstance, vector<T> &v_targetValue,
									string strFileName, int nNumofFeatures, int nNumofInstance, bool bUseDense)
{
	ifstream readIn;
	readIn.open(strFileName.c_str());
	assert(readIn.is_open());
	vector<KeyValue> vSample;

	//for storing character from file
	int j = 0;
	string str;
//	int nMissingCount = 0;

	//get a sample
	char cColon;
	uint numFeaValue = 0;
	uint maxFeaId = 0;
	do {
		getline(readIn, str);
        if (str == "") break;
		istringstream in(str);
		int i = 0;
//		bool bMiss = false;
		T fValue = 0;
		in >> fValue;
		v_targetValue.push_back(fValue);

		//get features of a sample
		int nFeature;
		real x = 0xffffffff;
		while (in >> nFeature >> cColon >> x)
		{
			if(nFeature > maxFeaId)
				maxFeaId = nFeature;
			numFeaValue++;
			//assert(x > 0 && x <= 1);
			assert(cColon == ':');
			if(bUseDense == true)
			{
				while(int(vSample.size()) < nFeature - 1 && int(vSample.size()) < nNumofFeatures)
				{
					Push(i, 0, vSample);
					i++;
				}
			}

			if(nNumofFeatures == int(vSample.size()))
			{
				break;
			}
			assert(int(vSample.size()) <= nNumofFeatures);
			if(bUseDense == true)
				assert(i == nFeature - 1);

			Push(nFeature - 1, x, vSample);
			i++;
		}
		//skip an empty line (usually this case happens in the last line)
//		if(x == 0xffffffff)
//			continue;
		//fill the value of the rest of the features as 0
		if(bUseDense == true)
		{
			while(int(vSample.size()) < nNumofFeatures)
			{
				Push(i, 0, vSample);
				i++;
			}
		}
		j++;
		v_vInstance.push_back(vSample);

		//clear vector
		vSample.clear();
	} while (readIn.eof() != true && j < nNumofInstance);//nNumofInstance is to enable reading a subset.

	//clean eof bit, when pointer reaches end of file
	if(readIn.eof())
	{
		readIn.clear();
	}
    printf("# of instances: %d;  max feature id2: %d; # of fvalue: %d\n", v_vInstance.size(), maxFeaId, numFeaValue);
}



#endif /* TRAININGDATAIO_H_ */
