/**
 * trainingDataIO.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef DATAIO_H_
#define DATAIO_H_

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "../../svm-shared/my_assert.h"
#include "../../svm-shared/constant.h"
#include "../../svm-shared/gpu_global_utility.h"
#include "../../SharedUtility/KeyValue.h"

using std::string;
using std::vector;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

class CDataIOOps {
public:
    CDataIOOps() {}

    virtual ~CDataIOOps() {}

    bool
    ReadFromFile(string strFileName, int nNumofFeature, vector<vector<float_point> > &v_vData, vector<int> &v_nLabel);

    virtual bool ReadFromFile(string strFileName, vector<vector<float_point> > &v_vData, vector<int> &v_nLabel) {
        return false;
    }

    bool OrganizeSamples(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
                         vector<vector<float_point> > &v_vAllSample, vector<int> &v_nLabel);

    bool ReadFromFileSparse(string strFileName, int nNumofFeature, vector<vector<KeyValue> > &v_vSampleData,
                            vector<int> &v_nLabel);
};

class CReadHelper {
public:
    static void Randomize(vector<vector<float_point> > &v_vPos, vector<vector<float_point> > &v_vNeg);

    static void
    ReadLibSVMDataFormat(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
                         string strFileName, int nNumofFeatures, int nNumofSamples = -1);

    static void
    ReadMultiClassData(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
                       string strFileName, int nNumofFeatures, int nNumofSamples);

    static void
    ReadLibSVMMultiClassData(vector<vector<float_point> > &v_vSamples, vector<int> &v_nLabels, string strFileName,
                             long nNumofFeatures);

    static void ReadLibSVMMultiClassDataSparse(vector<vector<KeyValue> > &v_vSamples, vector<int> &v_nLabels,
                                               string strFileName, long nNumofFeatures);
};


#endif /* DATAIO_H_ */
