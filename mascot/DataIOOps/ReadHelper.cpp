/*
 * ReadHelper.cpp
 *
 *  Created on: 26/04/2013
 *  Author: Zeyi Wen
 */

#include <sstream>
#include <limits>
#include "DataIO.h"
#include "../../DataReader/LibsvmReaderSparse.h"

using std::istringstream;

/*
 * @brief: randomise
 */
void CReadHelper::Randomize(vector<vector<float_point> > &v_vPos, vector<vector<float_point> > &v_vNeg)
{
    vector<vector<float_point> > v_vTempPos;
    vector<vector<float_point> > v_vTempNeg;

    srand(1000);
    while(!v_vPos.empty())
    {
        int nIndex = rand() % v_vPos.size();
        v_vTempPos.push_back(v_vPos[nIndex]);
        vector<vector<float_point> >::iterator it = v_vPos.begin();
        v_vPos.erase(it + nIndex);
    }

    while(!v_vNeg.empty())
    {
        int nIndex = rand() % v_vNeg.size();
        v_vTempNeg.push_back(v_vNeg[nIndex]);
        vector<vector<float_point> >::iterator it = v_vNeg.begin();
        v_vNeg.erase(it + nIndex);
    }
    assert(v_vPos.empty() && v_vPos.empty());
    v_vPos = v_vTempPos;
    v_vNeg = v_vTempNeg;
}

/**
 * @brief: read libsvm format 2-class data and store in dense format. For reading the whole set or a subset.
 */
void CReadHelper::ReadLibSVMDataFormat(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
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

    vector<float_point> vSample;

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
        float_point x;
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
void CReadHelper::ReadMultiClassData(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
        string strFileName, int nNumofFeatures, int nNumofSamples)
{
    ifstream readIn;
    readIn.open(strFileName.c_str());
    assert(readIn.is_open());
    vector<float_point> vSample;

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
        float_point x;
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

/**
 * @brief: read libsvm format data and store in dense form.
 */
void CReadHelper::ReadLibSVMMultiClassData(vector<vector<float_point> > &v_vSamples, vector<int> &v_nLabels,
										   string strFileName, long nNumofFeatures){
	LibSVMDataReader drHelper;
	drHelper.ReadLibSVMAsDense(v_vSamples, v_nLabels, strFileName, nNumofFeatures);

	/*
    ifstream readIn;
    readIn.open(strFileName.c_str());
    assert(readIn.is_open());
    vector<float_point> vSample;
    int j = 0;
    string str;
    //get a sample
    char cColon;
    while (!readIn.eof()) {
        j++;
        getline(readIn, str);
        if (str == "") break;
        istringstream in(str);
        int i = 0;
        int nLabel = 0;
        in >> nLabel;

        //get features of a sample
        int nFeature;
        float_point x;
        while (in >> nFeature >> cColon >> x) {
            i++;
            //assert(x >= -1 && x <= 1);
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
        v_vSamples.push_back(vSample);
        v_nLabels.push_back(nLabel);
        //clear vector
        vSample.clear();
    };

    //clean eof bit, when pointer reaches end of file
    if(readIn.eof())
    {
        //cout << "end of file" << endl;
        readIn.clear();
    }*/
}

/**
 * @brief: read libsvm format data and store in sparse form.
 */
void CReadHelper::ReadLibSVMMultiClassDataSparse(vector<vector<svm_node> > &v_vInstance, vector<int> &v_nLabels,
												 const string strFileName, const long nNumofFeatures){
    ifstream readIn;
    readIn.open(strFileName.c_str());
    assert(readIn.is_open());
    vector<svm_node> vIns;
    int j = 0;
    string str;
    //get a sample
    char cColon;
    while (!readIn.eof()) {
        j++;
        getline(readIn, str);
        if (str == "") break;
        istringstream in(str);
        int i = 0;
        int nLabel = 0;
        in >> nLabel;

        //get features of a sample
        int featureId;
        float_point x;
        while (in >> featureId >> cColon >> x) {
            i++;
            assert(featureId <= nNumofFeatures && cColon == ':');
            vIns.push_back(svm_node(featureId,x));
        }
        vIns.push_back(svm_node(-1,0));

        v_vInstance.push_back(vIns);
        v_nLabels.push_back(nLabel);
        //clear vector
        vIns.clear();
    };

    //clean eof bit, when pointer reaches end of file
    if(readIn.eof())
    {
        readIn.clear();
    }
}
