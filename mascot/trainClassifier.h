/*
 * testTrainer.h
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi Wen
 */

#ifndef TESTTRAINER_H_
#define TESTTRAINER_H_

#include <iostream>
#include "svmModel.h"
#include "../svm-shared/svmParam.h"
#include "../svm-shared/gpu_global_utility.h"

using std::string;

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature, SvmModel &model, bool evaluateTrainingError = false);

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature);

void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel, vector<real> &classificationError);

void trainOVASVM(SVMParam &param, string strTrainingFileName, int numFeature,  bool evaluteTrainingError, string strTestingFileName);

float evaluateOVASVMClassifier(SvmModel &model, vector<vector<int> > &combPredictLabels, string strTrainingFileName, int numFeature);

float evaluateOVABinaryClassifier(vector<vector<int> > &combPredictLabels, SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
              vector<real> &classificationError);
void evaluateOVA(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<int> > &combPredictLabels, vector<int> &originalPositivelabel, float testingTime);

#endif /* TESTTRAINER_H_ */
