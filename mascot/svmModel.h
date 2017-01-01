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
public:
    vector<int> label;
    unsigned int nrClass;

private:
    SVMParam param;
    unsigned int cnr2;					//total number of svm models to train
    int numOfSVs;
    int numOfFeatures;
    vector<vector<int> > svIndex;
    vector<vector<svm_node> > svMap;
    CSRMatrix *svMapCSRMat = NULL;
    vector<vector<float_point> > coef;
    vector<int> start;					//for multiclass, start position for each class of instances
    vector<int> count;					//support vectors of the i-th class
    vector<float_point> rho;
    vector<float_point> probA;
    vector<float_point> probB;
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

	//have changed the type of *dec_values,& A,& B
//	void gpu_sigmoid_train(int l, const float_point *dec_values, const float_point *labels,
//	float_point& A, float_point& B);

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

    void
    addBinaryModel(const SvmProblem &subProblem, const vector<int> &svIndex, const vector<float_point> &coef,
                   float_point rho, int i,
                   int j);

    bool isProbability() const;
};

#endif //MASCOT_SVM_SVMMODEL_H
