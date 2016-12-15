//
// Created by ss on 16-12-14.
//

#ifndef MASCOT_SVM_MULTISMOSOLVER_H
#define MASCOT_SVM_MULTISMOSOLVER_H


#include "svmProblem.h"
#include "svmModel.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/baseHessian.h"

class MultiSmoSolver {
public:
    MultiSmoSolver(const SvmProblem &problem, SvmModel &model, const SVMParam &param) :
            problem(problem), model(model), param(param) {}

    void solve();


private:
    const SvmProblem &problem;
    SvmModel &model;
    const SVMParam &param;

    void initCache(int cacheSize);
    CCache *gpuCache;
    BaseHessian *hessianCalculator;

    void init4Training(const SvmProblem &subProblem);

    bool iterate(SvmProblem &subProblem);
    void getHessianRow(int rowIndex, float_point *devHessianRow);
    void updateTwoWeight(float_point fMinLowValue, float_point fMinValue,
								 int nHessianRowOneInMatrix, int nHessianRowTwoInMatrix,
								 float_point fKernelValue,
								 float_point &fY1AlphaDiff, float_point &fY2AlphaDiff, const int *label);

    void extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<float_point> &coef, float_point &rho) const;
    void deinit4Training();

    float_point *devAlpha;
    vector<float_point> alpha;
    float_point *devYiGValue;
    int *devLabel;

    float_point *devBlockMin;
    int *devBlockMinGlobalKey;
    float_point *devBlockMinYiFValue;
    float_point *devMinValue;
    int *devMinKey;
    float_point upValue;
    float_point lowValue;
    float_point *devBuffer;
    float_point *hostBuffer;

    float_point *hessianRow;
    float_point *devHessianSampleRow1;
    float_point *devHessianSampleRow2;
    float_point *devHessianMatrixCache;
    float_point *devHessianDiag;
    float_point *hessianDiag;
    dim3 gridSize;
    int numOfBlock;

};


#endif //MASCOT_SVM_MULTISMOSOLVER_H
