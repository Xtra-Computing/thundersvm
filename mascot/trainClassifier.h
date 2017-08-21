/*
 * testTrainer.h
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi Wen
 */

#ifndef TESTTRAINER_H_
#define TESTTRAINER_H_

#include <iostream>
#include <fstream>
#include "svmModel.h"
#include "../svm-shared/svmParam.h"
#include "../svm-shared/gpu_global_utility.h"
using namespace std;
using std::string;

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature, SvmModel &model, bool evaluateTrainingError = false);

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature);

void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel, vector<real> &classificationErrori, ofstream &ofs);

void trainOVASVM(SVMParam &param, string strTrainingFileName, int numFeature,  bool evaluteTrainingError, string strTestingFileNamei, ofstream &ofs);

float evaluateOVABinaryClassifier(vector<real>  &combDecValue, vector<vector<int> > &combPredictLabels, SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
              vector<real> &classificationError);
void evaluateOVAVote(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<int> > &combPredictLabels, vector<int> &originalPositivelabel, float testingTime, ofstream &ofs);

void evaluateOVADecValue(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<real> > &combDecValue, vector<int> originalPositiveLabel, float testingTime);


#endif /* TESTTRAINER_H_ */
