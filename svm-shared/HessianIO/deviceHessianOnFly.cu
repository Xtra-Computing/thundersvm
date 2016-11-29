//
// Created by ss on 16-11-15.
//

#include "deviceHessianOnFly.h"
#include "../constant.h"

__global__ void RBFKernel(const float_point *aSelfDot, int bRow, float_point *dotProduct, int numOfSamples,
                          float gamma) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numOfSamples) {
        const float bSelfDot = aSelfDot[bRow];
        dotProduct[idx] = expf(-(aSelfDot[idx] + bSelfDot - dotProduct[idx] * 2) * gamma);
    }
}

void DeviceHessianOnFly::ReadRow(int nPosofRowAtHessian, float_point *pfHessianRow) {
//    printf("READ ROW %d\n", nPosofRowAtHessian);
    const int nnzB = problem.getCSRRowPtr()[nPosofRowAtHessian + 1] - problem.getCSRRowPtr()[nPosofRowAtHessian];
    const float_point *devBVal = devValA + problem.getCSRRowPtr()[nPosofRowAtHessian];
    const int *devBInd = devColIndA + problem.getCSRRowPtr()[nPosofRowAtHessian];
    float_point *devBDense;
    checkCudaErrors(cudaMalloc((void **) &devBDense, sizeof(float_point) * problem.getNumOfFeatures()));
    checkCudaErrors(cudaMemset(devBDense, 0, sizeof(float_point) * problem.getNumOfFeatures()));
    cusparseSsctr(handle, nnzB, devBVal, devBInd, devBDense, CUSPARSE_INDEX_BASE_ZERO);
    float_point *devC;
    checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * problem.getNumOfSamples()));
    checkCudaErrors(cudaMemset(devC,0,sizeof(float_point) * problem.getNumOfSamples()));
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   problem.getNumOfSamples(), 1, problem.getNumOfFeatures(),
                   nnz, &one, descr, devValA, devRowPtrA, devColIndA,
                   devBDense, problem.getNumOfFeatures(), &zero,
                   devC, problem.getNumOfSamples());
    RBFKernel << < Ceil(problem.getNumOfSamples(), BLOCK_SIZE), BLOCK_SIZE >> >
                                                                (devValASelfDot, nPosofRowAtHessian, devC, problem.getNumOfSamples(), gamma);
    checkCudaErrors(
            cudaMemcpy(pfHessianRow, devC, sizeof(float_point) * problem.getNumOfSamples(), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(devBDense));
    checkCudaErrors(cudaFree(devC));
}

bool DeviceHessianOnFly::PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                         vector<vector<float_point> > &v_v_DocVector) {
    return true;
}

bool DeviceHessianOnFly::GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples,
                                      float_point *pfHessianDiag) {
    for (int i = 0; i < nNumofTraingSamples; ++i) {
        pfHessianDiag[i] = 1;
    }
    return true;
}

//bool HostHessianOnFly::AllocateBuffer(int nNumofRows) {
//    checkCudaErrors(cudaMallocHost((void**)&m_pfHessianRows,sizeof(float_point)*m_nTotalNumofInstance*nNumofRows));
////    return BaseHessian::AllocateBuffer(nNumofRows);
//    return true;
//}
//
//bool HostHessianOnFly::ReleaseBuffer() {
////    return BaseHessian::ReleaseBuffer();
//    checkCudaErrors(cudaFreeHost(m_pfHessianRows));
//    return true;
//}

