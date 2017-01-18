//
// Created by ss on 16-11-15.
//

#ifndef MASCOT_SVM_HOSTHESSIANONFLY_H
#define MASCOT_SVM_HOSTHESSIANONFLY_H


#include <cusparse.h>
#include <cuda.h>
#include "baseHessian.h"
#include "hostKernelCalculater/kernelFunction.h"
#include "hostKernelCalculater/rbfKernelFunction.h"
#include"cuda_runtime.h"
#include"helper_cuda.h"
#include "../gpu_global_utility.h"
#include "../../mascot/svmProblem.h"
#include "../csrMatrix.h"

class DeviceHessianOnFly : public BaseHessian {
public:
    DeviceHessianOnFly(const SvmProblem &subProblem, float_point gamma);

    ~DeviceHessianOnFly();

    void ReadRow(int nPosofRowAtHessian, float_point *devHessianRow, int start, int end);

    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<float_point> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag) override;

private:
    CSRMatrix csrMat;
    const float_point gamma;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    float_point *devVal;
    float_point *devValSelfDot;
    float_point *devDenseVector;
    vector<int> csrRowPtrSplit;
    int *devRowPtr;
    int *devRowPtrSplit;
    int *devColInd;
    float_point one;
    float_point zero;

};


#endif //MASCOT_SVM_HOSTHESSIANONFLY_H
