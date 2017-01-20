/*
 * evaluateClassifier.h
 *
 *  Created on: 21 Jan 2017
 *      Author: Zeyi Wen
 */

#ifndef MASCOT_CLASSIFIEREVALUATER_H_
#define MASCOT_CLASSIFIEREVALUATER_H_

#include <vector>
#include <stdio.h>
#include "../SharedUtility/DataType.h"

using std::vector;

class ClassifierEvaluater{
public:
	static vector<float_point> trainingError;
	static vector<float_point> testingError;
public:

	static void evaluateSubClassifier(const vector<vector<int> > &missLabellingMatrix, vector<float_point> &vErrorRate);
	static vector<float_point> updateC(const vector<float_point> &vOldC);
};



#endif /* MASCOT_CLASSIFIEREVALUATER_H_ */
