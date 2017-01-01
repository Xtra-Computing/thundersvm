/*
 * multiProdictor.h
 *
 *  Created on: 1 Jan 2017
 *      Author: Zeyi Wen
 */

#ifndef MASCOT_MULTIPREDICTOR_H_
#define MASCOT_MULTIPREDICTOR_H_

#include <vector>
#include "../svm-shared/host_constant.h"
#include "../svm-shared/gpu_global_utility.h"
#include "svmModel.h"
#include "svmParam.h"

using std::vector;

class MultiPredictor{
private:
    const SvmModel &model;
    const SVMParam &param;

public:
	MultiPredictor(const SvmModel &model, const SVMParam &param):model(model), param(param){}
	~MultiPredictor(){}

    vector<int> predict(const vector<vector<svm_node> > &v_vSamples, bool probability) const;

    vector<vector<float_point> > predictProbability(const vector<vector<svm_node> > &) const;

    void predictValues(const vector<vector<svm_node> > &, vector<vector<float_point> > &) const;

private:
    float_point sigmoidPredict(float_point decValue, float_point A, float_point B) const;

    void multiClassProbability(const vector<vector<float_point> > &, vector<float_point> &) const;
};



#endif /* MASCOT_MULTIPREDICTOR_H_ */
