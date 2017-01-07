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
#include "../../mascot/csrMatrix.h"

class DeviceHessianOnFly : public BaseHessian {
public:
    DeviceHessianOnFly(const SvmProblem &subProblem, float_point gamma);

    ~DeviceHessianOnFly() {
        checkCudaErrors(cudaFree(devVal));
        checkCudaErrors(cudaFree(devValSelfDot));
        checkCudaErrors(cudaFree(devRowPtr));
        checkCudaErrors(cudaFree(devRowPtrSplit));
        checkCudaErrors(cudaFree(devColInd));
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);
    }


    void ReadRow(int nPosofRowAtHessian, float_point *devHessianRow, int start, int end);
    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<float_point> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag) override;

//    virtual bool AllocateBuffer(int nNumofRows) override;

//    virtual bool ReleaseBuffer() override;

private:
//    const SvmProblem &problem;
//    const vector<vector<svm_node> > &samples;
//    const int numOfFeatures;
    CSRMatrix csrMat;
    const float_point gamma;
    //TODO move initializing handle and descr outside
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    float_point *devVal;
    float_point *devValSelfDot;
    vector<int> csrRowPtrSplit;
    int *devRowPtr;
    int *devRowPtrSplit;
    int *devColInd;
    float_point one;
    float_point zero;

};


#endif //MASCOT_SVM_HOSTHESSIANONFLY_H
