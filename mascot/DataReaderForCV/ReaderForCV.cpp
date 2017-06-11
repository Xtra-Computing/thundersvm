/*
 * ReadHelper.cpp
 *
 *  Created on: 26/04/2013
 *  Author: Zeyi Wen
 */

#include <stdlib.h>
#include <sstream>
#include <limits>
#include <fstream>
#include "../../svm-shared/my_assert.h"
#include "../DataReaderForCV/ReaderForCV.h"

using std::istringstream;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

/*
 * @brief: uniformly distribute positive and negative samples
 */
bool CReadForCrossValidation::OrganizeSamples(vector<vector<real> > &v_vPosSample, vector<vector<real> > &v_vNegSample,
                                 vector<vector<real> > &v_vAllSample, vector<int> &v_nLabel) {
    //merge two sets of samples into one
    int nSizeofPSample = v_vPosSample.size();
    int nSizeofNSample = v_vNegSample.size();
    double dRatio = ((double) nSizeofPSample) / nSizeofNSample;

    //put samples in a uniform way. This is to avoid the training set only having one class, during n-fold-cross-validation
    int nNumofPosInEachPart = 0;
    int nNumofNegInEachPart = 0;
    if (dRatio < 1) {
        nNumofPosInEachPart = 1;
        nNumofNegInEachPart = int(1.0 / dRatio);
    } else {
        nNumofPosInEachPart = (int) dRatio;
        nNumofNegInEachPart = 1;
    }

    vector<vector<real> >::iterator itPositive = v_vPosSample.begin();
    vector<vector<real> >::iterator itNegative = v_vNegSample.begin();
    int nCounter = 0;
    while (itPositive != v_vPosSample.end() || itNegative != v_vNegSample.end()) {
        for (int i = 0; i < nNumofPosInEachPart && itPositive != v_vPosSample.end(); i++) {
            nCounter++;
            v_vAllSample.push_back(*itPositive);
            v_nLabel.push_back(1);
            itPositive++;
        }

        for (int i = 0; i < nNumofNegInEachPart && itNegative != v_vNegSample.end(); i++) {
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

/*
 * @brief: randomise
 */
void CReadForCrossValidation::Randomize(vector<vector<real> > &v_vPos, vector<vector<real> > &v_vNeg)
{
    vector<vector<real> > v_vTempPos;
    vector<vector<real> > v_vTempNeg;

    srand(1000);
    while(!v_vPos.empty())
    {
        int nIndex = rand() % v_vPos.size();
        v_vTempPos.push_back(v_vPos[nIndex]);
        vector<vector<real> >::iterator it = v_vPos.begin();
        v_vPos.erase(it + nIndex);
    }

    while(!v_vNeg.empty())
    {
        int nIndex = rand() % v_vNeg.size();
        v_vTempNeg.push_back(v_vNeg[nIndex]);
        vector<vector<real> >::iterator it = v_vNeg.begin();
        v_vNeg.erase(it + nIndex);
    }
    assert(v_vPos.empty() && v_vPos.empty());
    v_vPos = v_vTempPos;
    v_vNeg = v_vTempNeg;
}

/**
 * @brief: read libsvm format 2-class data and store in dense format. For reading the whole set or a subset.
 */
void CReadForCrossValidation::ReadLibSVMDataFormat(vector<vector<real> > &v_vPosSample, vector<vector<real> > &v_vNegSample,
        							   string strFileName, int nNumofFeatures, int nNumofSamples)
{
	if(nNumofSamples == -1){//read the whole dataset
		nNumofSamples = std::numeric_limits<int>::max();
	}
    ifstream readIn;
    readIn.open(strFileName.c_str());
    assert(readIn.is_open());
    if(!readIn.is_open())
    {
        cerr << "open file \"" << strFileName << "\" failed"  << endl;
        exit(0);
    }

    vector<real> vSample;

    //for storing character from file
    int j = 0;
    string str;
    int nMissingCount = 0;

    //get a sample
    char cColon;
    do {
        j++;
        getline(readIn, str);

        istringstream in(str);
        int i = 0;
        bool bMiss = false;
        int nLabel = 0;
        in >> nLabel;
        assert(nLabel == -1 || nLabel == 1);

        //get features of a sample
        int nFeature;
        real x;
        while (in >> nFeature >> cColon >> x) {
            i++;
            assert(x > 0 && x <= 1);
            assert(nFeature <= nNumofFeatures && cColon == ':');
            while(vSample.size() < nFeature - 1)
            {
                if(vSample.size() == nNumofFeatures)
                    break;
                vSample.push_back(0);
            }
            if(vSample.size() == nNumofFeatures)
                break;
            vSample.push_back(x);
            if(vSample.size() == nNumofFeatures)
                break;
            assert(vSample.size() <= nNumofFeatures);
        }
        //fill the value of the rest of the features as 0
        while(vSample.size() < nNumofFeatures)
        {
            vSample.push_back(0);
        }

        if (nLabel == -1)
        {
            v_vNegSample.push_back(vSample);
        }
        else if(nLabel == 1)
        {
            v_vPosSample.push_back(vSample);
        }
        //clear vector
        vSample.clear();
    } while (readIn.eof() != true && j < nNumofSamples);///72309 is the number of samples

    //clean eof bit, when pointer reaches end of file
    if(readIn.eof())
    {
        //cout << "end of file" << endl;
        readIn.clear();
    }
}

/**
 * @brief: read multiclass data and convert to 2-class data by even and odd.
 */
void CReadForCrossValidation::ReadMultiClassData(vector<vector<real> > &v_vPosSample, vector<vector<real> > &v_vNegSample,
        string strFileName, int nNumofFeatures, int nNumofSamples)
{
    ifstream readIn;
    readIn.open(strFileName.c_str());
    assert(readIn.is_open());
    vector<real> vSample;

    //for storing character from file
    int j = 0;
    string str;
    int nMissingCount = 0;

    //get a sample
    char cColon;
    do {
        j++;
        getline(readIn, str);

        istringstream in(str);
        int i = 0;
        bool bMiss = false;
        int nLabel = 0;
        in >> nLabel;
        assert(nLabel >= 0 && nLabel <= 10);

        //get features of a sample
        int nFeature;
        real x;
        while (in >> nFeature >> cColon >> x) {
            i++;
            assert(x >= -1 && x <= 1);
            assert(nFeature <= nNumofFeatures && cColon == ':');
            while(vSample.size() < nFeature - 1)
            {
                vSample.push_back(0);
            }
            vSample.push_back(x);
            assert(vSample.size() <= nNumofFeatures);
        }
        //fill the value of the rest of the features as 0
        while(vSample.size() < nNumofFeatures)
        {
            vSample.push_back(0);
        }

        if (nLabel % 2 == 0)
        {
            v_vNegSample.push_back(vSample);
        }
        else
        {
            v_vPosSample.push_back(vSample);
        }
        //clear vector
        vSample.clear();
    } while (readIn.eof() != true && j < nNumofSamples);

    //clean eof bit, when pointer reaches end of file
    if(readIn.eof())
    {
        //cout << "end of file" << endl;
        readIn.clear();
    }
}

