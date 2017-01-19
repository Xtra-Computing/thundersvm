/**
 * trainingDataIO.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 **/

#ifndef DATAIO_H_
#define DATAIO_H_

#include <iostream>
#include <vector>
#include "../../SharedUtility/DataType.h"

using std::string;
using std::vector;


class CReadForCrossValidation {
public:
    static bool OrganizeSamples(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
                         vector<vector<float_point> > &v_vAllSample, vector<int> &v_nLabel);
    static void Randomize(vector<vector<float_point> > &v_vPos, vector<vector<float_point> > &v_vNeg);

    static void
    ReadLibSVMDataFormat(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
                         string strFileName, int nNumofFeatures, int nNumofSamples = -1);

    static void
    ReadMultiClassData(vector<vector<float_point> > &v_vPosSample, vector<vector<float_point> > &v_vNegSample,
                       string strFileName, int nNumofFeatures, int nNumofSamples);

};

#endif /* DATAIO_H_ */
