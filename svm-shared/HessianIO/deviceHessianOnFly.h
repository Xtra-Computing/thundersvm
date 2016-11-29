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
    DeviceHessianOnFly(SvmProblem &problem, float_point gamma) :
            gamma(gamma), problem(problem), zero(0.0f), one(1.0f) {
        cusparseCreate(&handle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        problem.convert2CSR();
        nnz = problem.getNnz();
        checkCudaErrors(cudaMalloc((void **) &devValA, sizeof(float_point) * problem.getNnz()));
        checkCudaErrors(cudaMalloc((void **) &devValASelfDot, sizeof(float_point) * problem.getNnz()));
        checkCudaErrors(cudaMalloc((void **) &devRowPtrA, sizeof(int) * (problem.getNumOfSamples() + 1)));
        checkCudaErrors(cudaMalloc((void **) &devColIndA, sizeof(int) * (problem.getNnz())));
        checkCudaErrors(cudaMemcpy(devValA, problem.getCSRVal(), sizeof(float_point) * problem.getNnz(),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devValASelfDot, problem.getCSRValSelfDot(),
                                   sizeof(float_point) * problem.getNumOfSamples(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devRowPtrA, problem.getCSRRowPtr(), sizeof(int) * (problem.getNumOfSamples() + 1),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(devColIndA, problem.getCSRColInd(), sizeof(int) * (problem.getNnz()),
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

    virtual void ReadRow(int nPosofRowAtHessian, float_point *pfHessianRow) override;

    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<float_point> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag) override;

//    virtual bool AllocateBuffer(int nNumofRows) override;

//    virtual bool ReleaseBuffer() override;

private:
    SvmProblem &problem;
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
