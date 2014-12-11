/**
 * trainingDataIO.cu
 * @brief: this file includes the definition of functions for reading data
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#include "DataIO.h"
#include <iostream>

using std::cout;
using std::endl;

bool CDataIOOps::ReadFromFile(string strFileName, int nNumofFeature, vector<vector<float_point> > &v_vSampleData, vector<int> &v_nLabel)
{
	bool nReturn = true;
	v_nLabel.clear();
	//these two containers is for storing positive and negative samples from file respectively
	vector<vector<float_point> > v_vPosSample;
	vector<vector<float_point> > v_vNegSample;

	cout << "reading data..." << endl;
	//read data from file
	CReadHelper::ReadLibSVMDataFormat(v_vPosSample, v_vNegSample, strFileName, nNumofFeature);
	//organize the samples
	cout << "Dataset has " << v_vPosSample.size() << " positive examples; " << v_vNegSample.size() << " negative examples." << endl;
	int nNumofExamples = v_vPosSample.size() + v_vNegSample.size();

	v_vSampleData.reserve(nNumofExamples);
	v_nLabel.reserve(nNumofExamples);
	if(v_vPosSample.size() > 0 && v_vNegSample.size() > 0)
		OrganizeSamples(v_vPosSample, v_vNegSample, v_vSampleData, v_nLabel);
	else if(v_vPosSample.size() > 0)
	{
		for(int i = 0; i < int(v_vPosSample.size()); i++)
			v_nLabel.push_back(1);
		v_vSampleData = v_vPosSample;
	}
	else
	{
		for(int i = 0; i < int(v_vNegSample.size()); i++)
			v_nLabel.push_back(-1);
		v_vSampleData = v_vNegSample;
	}
//	cout << "sample organised" << endl;

	assert(v_vSampleData.size() == nNumofExamples);

	return nReturn;
}

/*
 * @brief: uniformly distribute positive and negative samples
 */
bool CDataIOOps::OrganizeSamples(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
										 vector<vector<float_point> > &v_vAllSample, vector<int> &v_nLabel)
{
	//merge two sets of samples into one
	int nSizeofPSample = v_vPosSample.size();
	int nSizeofNSample = v_vNegSample.size();
	double dRatio = ((double)nSizeofPSample) / nSizeofNSample;

	//put samples in a uniform way. This is to avoid the training set only having one class, during n-fold-cross-validation
	int nNumofPosInEachPart = 0;
	int nNumofNegInEachPart = 0;
	if(dRatio < 1)
	{
		nNumofPosInEachPart = 1;
		nNumofNegInEachPart = int(1.0 / dRatio);
	}
	else
	{
		nNumofPosInEachPart = (int)dRatio;
		nNumofNegInEachPart = 1;
	}

	vector<vector<float_point> >::iterator itPositive = v_vPosSample.begin();
	vector<vector<float_point> >::iterator itNegative = v_vNegSample.begin();
	int nCounter = 0;
	while(itPositive != v_vPosSample.end() || itNegative != v_vNegSample.end())
	{
		for(int i = 0; i < nNumofPosInEachPart && itPositive != v_vPosSample.end(); i++)
		{
			nCounter++;
			v_vAllSample.push_back(*itPositive);
			v_nLabel.push_back(1);
			itPositive++;
		}

		for(int i = 0; i < nNumofNegInEachPart && itNegative != v_vNegSample.end(); i++)
		{
			nCounter++;
			v_vAllSample.push_back(*itNegative);
			v_nLabel.push_back(-1);
			itNegative++;
		}
	}
	v_vPosSample.clear();
	v_vNegSample.clear();
	return true;
}



