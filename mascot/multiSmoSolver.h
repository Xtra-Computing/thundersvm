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

class MultiSmoSolver: public BaseSMO
{
public:
    MultiSmoSolver(const SvmProblem &problem, SvmModel &model, const SVMParam &param) :
            problem(problem), model(model), param(param),cache(problem, param) {
    }

    ~MultiSmoSolver(){
    };
    void solve();


private:
    const SvmProblem &problem;
    SvmModel &model;
    const SVMParam &param;

    void initCache(int cacheSize);
    CCache *gpuCache;
	GpuCache cache;
    DeviceHessianOnFly *hessianCalculator;

    void init4Training(const SvmProblem &subProblem);

    bool iterate(SvmProblem &subProblem);
    int getHessianRow(int rowIndex);
    void updateTwoWeight(float_point fMinLowValue, float_point fMinValue,
								 int nHessianRowOneInMatrix, int nHessianRowTwoInMatrix,
								 float_point fKernelValue,
								 float_point &fY1AlphaDiff, float_point &fY2AlphaDiff, const int *label);

    void extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<float_point> &coef, float_point &rho) const;
    void deinit4Training();

    virtual float_point *ObtainRow(int numTrainingInstance)
    {
    	cache.getHessianRow(IdofInstanceOne, devHessianInstanceRow1);
    	return devHessianInstanceRow1;
    }
    
    float_point *devMinValue;
    int *devMinKey;

    float_point *devHessianMatrixCache;
	int numOfElementEachRowInCache;

};


#endif //MASCOT_SVM_MULTISMOSOLVER_H
