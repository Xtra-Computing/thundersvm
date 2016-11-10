//
// Created by ss on 16-11-2.
//

#ifndef MASCOT_SVM_SVMMODEL_H
#define MASCOT_SVM_SVMMODEL_H

#include <vector>
#include "../svm-shared/gpu_global_utility.h"
#include "svmParam.h"
#include "svmProblem.h"

using std::vector;

class SvmModel {
private:
    SVMParam param;
    unsigned int nrClass;
    unsigned int cnr2;
    vector<vector<vector<float_point> > > supportVectors;
    vector<vector<float_point> > coef;
    vector<float_point> rho;
    vector<float_point> probA;
    vector<float_point> probB;
    vector<int> label;
    bool probability;

    unsigned int inline getK(int i, int j) const;

    void
    computeDecValues(float_point *devKernelValues, int numOfSamples, vector<float_point> &classificationResult, int) const;

    float_point *computeSVYiAlphaHessianSum(int nNumofTestSamples,
                                            float_point *pfDevSVYiAlphaHessian, const int &nNumofSVs,
                                            float_point fBias, float_point *pfFinalResult) const;

    void computeKernelValuesOnGPU(const float_point *devSamples, int numOfSamples,
                                      const vector<vector<float_point> > &supportVectors,
                                      float_point *devKernelValues) const;
    void addBinaryModel(const SvmProblem &, const svm_model &, int i, int j);

    float_point sigmoidPredict(float_point decValue, float_point A, float_point B) const;

    void multiClassProbability(const vector<vector<float_point> > &, vector<float_point> &) const;

    void
    sigmoidTrain(const float_point *decValues, const int, const vector<int> &labels, float_point &A, float_point &B);

public:

    void fit(const SvmProblem &problem, const SVMParam &param);

    vector<int> predict(const vector<vector<float_point> > &, bool probability=false) const;

    vector<vector<float_point> > predictProbability(const vector<vector<float_point> > &) const;

    void predictValues(const vector<vector<float_point> > &, vector<vector<float_point> > &) const;

    bool isProbability() const;
};


#endif //MASCOT_SVM_SVMMODEL_H
