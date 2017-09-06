//
// Created by ss on 16-12-14.
//

#ifndef MASCOT_SVM_MULTISMOSOLVER_H
#define MASCOT_SVM_MULTISMOSOLVER_H

#include "svmProblem.h"
#include "svmModel.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/baseHessian.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../svm-shared/Cache/gpuCache.h"
#include "../svm-shared/baseSMO.h"
#include "../svm-shared/svmParam.h"

class MultiSmoSolver: public BaseSMO
{
public:
    MultiSmoSolver(const SvmProblem &problem, SvmModel &model, const SVMParam &param);

    ~MultiSmoSolver();;
    void solve();

private:
    const SvmProblem &problem;
    SvmModel &model;
    const SVMParam &param;

    void init4Training(const SvmProblem &subProblem);


    void selectWorkingSetAndPreCompute(const SvmProblem &subProblem, uint numOfSelectPairs, real penaltyC);

    void extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<real> &coef, real &rho) const;
    real getObjValue(int numOfSamples) const;
    void deinit4Training();

    virtual real *ObtainRow(int numTrainingInstance)
    {
    	return devHessianInstanceRow1;
    }

    real *devAlphaDiff;
    int *devWorkingSet;
    int *devWorkingSetIndicator;
    real *devHessianMatrixCache;
    int nnz;
    real *devVal;
    int *devColInd;
    int *devRowPtr;
    real *devSelfDot;
    real *devFValue4Sort;
    int *devIdx4Sort;
    int workingSetSize;
    int q;
    vector<int> workingSet;
    vector<int> workingSetIndicator;
};


#endif //MASCOT_SVM_MULTISMOSOLVER_H
