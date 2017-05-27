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
    static bool OrganizeSamples(vector<vector<real> > &v_vPosSample, vector<vector<real> > &v_vNegSample,
                         vector<vector<real> > &v_vAllSample, vector<int> &v_nLabel);
    static void Randomize(vector<vector<real> > &v_vPos, vector<vector<real> > &v_vNeg);

    static void
    ReadLibSVMDataFormat(vector<vector<real> > &v_vPosSample, vector<vector<real> > &v_vNegSample,
                         string strFileName, int nNumofFeatures, int nNumofSamples = -1);

    static void
    ReadMultiClassData(vector<vector<real> > &v_vPosSample, vector<vector<real> > &v_vNegSample,
                       string strFileName, int nNumofFeatures, int nNumofSamples);

};

#endif /* DATAIO_H_ */
