/*
 * testTrainer.h
 *
 *  Created on: 31/10/2013
 *      Author: zeyi
 */

#ifndef TESTTRAINER_H_
#define TESTTRAINER_H_

#include "svmParam.h"
#include "../svm-shared/gpu_global_utility.h"
#include <iostream>

#include "svmModel.h"
using std::string;

SvmModel trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature);
svm_model trainBinarySVM(SvmProblem &problem, const SVMParam &param);
void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature);

#endif /* TESTTRAINER_H_ */
