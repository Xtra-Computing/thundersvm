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
    const SvmModel &model;
    const SVMParam &param;

public:
	MultiPredictor(const SvmModel &model, const SVMParam &param):model(model), param(param){}
	~MultiPredictor(){}

    vector<int> predict(const vector<vector<KeyValue> > &v_vSamples, bool probability) const;
private:
    vector<vector<float_point> > predictProbability(const vector<vector<KeyValue> > &) const;
    void computeDecisionValues(const vector<vector<KeyValue> > &, vector<vector<float_point> > &) const;

    float_point sigmoidPredict(float_point decValue, float_point A, float_point B) const;

    void multiClassProbability(const vector<vector<float_point> > &, vector<float_point> &) const;
};



#endif /* MASCOT_MULTIPREDICTOR_H_ */
