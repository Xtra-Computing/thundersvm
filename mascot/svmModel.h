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
    int numOfFeatures;
    int numOfSVs;
    vector<vector<vector<svm_node> > > supportVectors;
    vector<vector<float_point> > coef;
    vector<int> start;
    vector<int> count;
    vector<float_point> rho;
    vector<float_point> probA;
    vector<float_point> probB;
    vector<int> label;
    bool probability;

    //device pointers
    svm_node **devSVs = NULL;
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

    void
    sigmoidTrain(const float_point *decValues, const int, const vector<int> &labels, float_point &A, float_point &B);

    void transferToDevice();

    static void* trainWork(void *);
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

class WorkParam{
public:
    int i;
    int j;
    static CUcontext devContext;   //host threads must share the same device context in order to run device functions concurrently.
    cudaStream_t stream;
    SvmModel *model;
    const SvmProblem *problem;
    const SVMParam *param;
    WorkParam(int i, int j,cudaStream_t stream, SvmModel *model,const SvmProblem *problem, const SVMParam *param):
            i(i),j(j), stream(stream), model(model),problem(problem),param(param){};
    WorkParam(){};
};

#endif //MASCOT_SVM_SVMMODEL_H
