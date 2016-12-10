//
// Created by ss on 16-11-2.
//

#ifndef MASCOT_SVM_SVMMODEL_H
#define MASCOT_SVM_SVMMODEL_H

#include <vector>
#include <cstdio>
#include <driver_types.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "../svm-shared/gpu_global_utility.h"
#include "svmParam.h"
#include "svmProblem.h"
#include"pthread.h"
using std::vector;

class SvmModel {
private:
    SVMParam param;
    unsigned int nrClass;
    unsigned int cnr2;
    int numOfSVs;
    int numOfFeatures;
    vector<vector<int> > svIndex;
    vector<vector<svm_node> > svMap;
    CSRMatrix *svMapCSRMat = NULL;
    vector<vector<float_point> > coef;
    vector<int> start;
    vector<int> count;
    vector<float_point> rho;
    vector<float_point> probA;
    vector<float_point> probB;
    vector<int> label;
    bool probability;

    //device pointers
    float_point *devSVMapVal;
    float_point *devSVMapValSelfDot;
    int *devSVMapRowPtr;
    int *devSVMapColInd;
    int *devSVIndex;
//    svm_node **devSVs = NULL;
    float_point *devCoef = NULL;
    int *devStart = NULL;
    int *devCount = NULL;
    float_point *devRho = NULL;
    float_point *devProbA = NULL;
    float_point *devProbB = NULL;

    unsigned int inline getK(int i, int j) const;

    void addBinaryModel(const SvmProblem &, const svm_model &, int i, int j);

    float_point sigmoidPredict(float_point decValue, float_point A, float_point B) const;

    void multiClassProbability(const vector<vector<float_point> > &, vector<float_point> &) const;
	//have changed the type of *dec_values,& A,& B
	void gpu_sigmoid_train(int l, const float_point *dec_values, const float_point *labels, 
	float_point& A, float_point& B);

    void
    sigmoidTrain(const float_point *decValues, const int, const vector<int> &labels, float_point &A, float_point &B);

    void transferToDevice();

    //SvmModel has device pointer, so duplicating SvmModel is not allowed
    SvmModel &operator=(const SvmModel &);

    SvmModel(const SvmModel &);

public:
    ~SvmModel();

    SvmModel() {};

    void fit(const SvmProblem &problem, const SVMParam &param);

    vector<int> predict(const vector<vector<svm_node> > &, bool probability = false) const;

    vector<vector<float_point> > predictProbability(const vector<vector<svm_node> > &) const;

    void predictValues(const vector<vector<svm_node> > &, vector<vector<float_point> > &) const;

    bool isProbability() const;
};

#endif //MASCOT_SVM_SVMMODEL_H
