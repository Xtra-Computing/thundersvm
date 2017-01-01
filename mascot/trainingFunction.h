/*
 * testTrainer.h
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi Wen
 */

#ifndef TESTTRAINER_H_
#define TESTTRAINER_H_

#include "svmParam.h"
#include "../svm-shared/gpu_global_utility.h"
#include <iostream>

#include "svmModel.h"
#include "cuda_runtime.h"
using std::string;

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature, SvmModel &model, bool evaluateTrainingError = false);

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature);

#endif /* TESTTRAINER_H_ */
