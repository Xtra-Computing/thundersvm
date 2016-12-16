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

class DeviceHessianOnFly : public BaseHessian {
public:
    DeviceHessianOnFly(const SvmProblem &problem, float_point gamma) :
            gamma(gamma), problem(problem), zero(0.0f), one(1.0f), csrMat(problem.v_vSamples, problem.getNumOfFeatures()){
        cusparseCreate(&handle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        nnz = csrMat.getNnz();
        checkCudaErrors(cudaMalloc((void **) &devValA, sizeof(float_point) * csrMat.getNnz()));
        checkCudaErrors(cudaMalloc((void **) &devValASelfDot, sizeof(float_point) * csrMat.getNumOfSamples()));
        checkCudaErrors(cudaMalloc((void **) &devRowPtrA, sizeof(int) * (csrMat.getNumOfSamples() + 1)));
        checkCudaErrors(cudaMalloc((void **) &devColIndA, sizeof(int) * (csrMat.getNnz())));
        checkCudaErrors(cudaMemcpy(devValA,csrMat.getCSRVal(), sizeof(float_point) * csrMat.getNnz(),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devValASelfDot, csrMat.getCSRValSelfDot(),
                                   sizeof(float_point) * problem.getNumOfSamples(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devRowPtrA,csrMat.getCSRRowPtr(), sizeof(int) * (problem.getNumOfSamples() + 1),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devColIndA,csrMat.getCSRColInd(), sizeof(int) * (csrMat.getNnz()),
                                   cudaMemcpyHostToDevice));

    };

    ~DeviceHessianOnFly() {
        checkCudaErrors(cudaFree(devValA));
        checkCudaErrors(cudaFree(devValASelfDot));
        checkCudaErrors(cudaFree(devRowPtrA));
        checkCudaErrors(cudaFree(devColIndA));
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);
    }

    virtual void ReadRow(int nPosofRowAtHessian, float_point *devHessianRow) override;

    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<float_point> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag) override;

//    virtual bool AllocateBuffer(int nNumofRows) override;

//    virtual bool ReleaseBuffer() override;

private:
    const SvmProblem &problem;
    CSRMatrix csrMat;
    const float_point gamma;
    //TODO move initializing handle and descr outside
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;

    int nnz;
    float_point *devValA;
    float_point *devValASelfDot;
    int *devRowPtrA;
    int *devColIndA;
    float_point one;
    float_point zero;
};


#endif //MASCOT_SVM_HOSTHESSIANONFLY_H
