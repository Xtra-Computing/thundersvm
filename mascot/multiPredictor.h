/*
 * multiProdictor.h
 *
 *  Created on: 1 Jan 2017
 *      Author: Zeyi Wen
 */

#ifndef MASCOT_MULTIPREDICTOR_H_
#define MASCOT_MULTIPREDICTOR_H_

#include <vector>
#include "svmModel.h"
#include "../svm-shared/svmParam.h"
#include "../svm-shared/host_constant.h"
#include "../svm-shared/gpu_global_utility.h"
#include "../SharedUtility/KeyValue.h"

using std::vector;

class MultiPredictor{
private:
    SvmModel &model;
    const SVMParam &param;

public:
	MultiPredictor(SvmModel &model, const SVMParam &param):model(model), param(param){}
	~MultiPredictor(){}

    vector<int> predict(const vector<vector<KeyValue> > &v_vSamples, const vector<int> &vnOriginalLabel = vector<int>()) const;
	void computeDecisionValues(const vector<vector<KeyValue> > &, vector<vector<real> > &) const;
private:
    vector<vector<real> > predictProbability(const vector<vector<KeyValue> > &, const vector<int> &vnOriginalLabel) const;

    real sigmoidPredict(real decValue, real A, real B) const;

    void multiClassProbability(const vector<vector<real> > &, vector<real> &) const;
};



#endif /* MASCOT_MULTIPREDICTOR_H_ */
