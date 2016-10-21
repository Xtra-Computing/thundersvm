/*
 * cvFunction.h
 *
 *  Created on: 09/12/2014
 *      Author: Zeyi Wen
 */

#ifndef CVFUNCTION_H_
#define CVFUNCTION_H_

#include "svmParam.h"
#include <iostream>

using std::string;

void crossValidation(SVMParam &param, string strTrainingFileName);


#endif /* CVFUNCTION_H_ */
