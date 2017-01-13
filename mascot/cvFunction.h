/*
 * cvFunction.h
 *
 *  Created on: 09/12/2014
 *      Author: Zeyi Wen
 */

#ifndef CVFUNCTION_H_
#define CVFUNCTION_H_

#include <iostream>
#include "../svm-shared/svmParam.h"
#include "../svm-shared/gpu_global_utility.h"

using std::string;

void crossValidation(SVMParam &param, string strTrainingFileName);
void gridSearch(Grid &SGrid, string strTrainingFileName);

#endif /* CVFUNCTION_H_ */
