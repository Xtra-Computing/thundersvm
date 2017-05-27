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
    DeviceHessianOnFly(const SvmProblem &subProblem, real gamma);

    ~DeviceHessianOnFly();

    void ReadRow(int nPosofRowAtHessian, real *devHessianRow, int start, int end);

    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<real> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, real *pfHessianDiag) override;

private:
    CSRMatrix csrMat;
    const real gamma;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    real *devVal;
    real *devValSelfDot;
    real *devDenseVector;
    vector<int> csrRowPtrSplit;
    int *devRowPtr;
    int *devRowPtrSplit;
    int *devColInd;
    real one;
    real zero;

};


#endif //MASCOT_SVM_HOSTHESSIANONFLY_H
