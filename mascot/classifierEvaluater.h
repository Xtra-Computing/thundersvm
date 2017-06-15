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
#include "svmModel.h"

using std::vector;

class ClassifierEvaluater{
public:
	static vector<real> trainingError;
	static vector<real> testingError;
public:

	static void collectSubSVMInfo(SvmModel &model, int insId, int trueLable, int nrClass, const vector<vector<real> > &predictedRes, bool isProbabilistic);
	static void evaluateSubClassifier(const vector<vector<int> > &missLabellingMatrix, vector<real> &vErrorRate);
	static vector<real> updateC(const vector<real> &vOldC);
};



#endif /* MASCOT_CLASSIFIEREVALUATER_H_ */
