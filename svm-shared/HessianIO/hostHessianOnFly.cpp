//
// Created by ss on 16-11-15.
//

#include "deviceHessianOnFly.h"
#include "../constant.h"

__global__ void RBFKernel(const float_point *aSelfDot, const float_point bSelfDot, float_point *dotProduct, int numOfSamples,
                          float gamma){
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numOfSamples){
        dotProduct[idx] = expf((aSelfDot[idx] + bSelfDot - dotProduct[idx] * 2)*);
    }
}
void HostHessianOnFly::ReadRow(int nPosofRowAtHessian, float_point *pfHessianRow) {
//    kernelCalculator.ComputeSparseRow(samples,nPosofRowAtHessian,1,pfHessianRow);
    const int nnzB = problem.getCSRRowPtr()[nPosofRowAtHessian+1] - problem.getCSRRowPtr()[nPosofRowAtHessian];
    const float_point *devBVal = devValA + problem.getCSRRowPtr()[nPosofRowAtHessian];
    const int *devBInd = devColIndA + problem.getCSRRowPtr()[nPosofRowAtHessian];
    float_point *devBDense;
    checkCudaErrors(cudaMalloc((void**)&devBDense,sizeof(float_point) * problem.getNumOfFeatures()));
    cusparseSsctr(handle,nnzB,devBVal,devBInd,devBDense,CUSPARSE_INDEX_BASE_ZERO);
    cusparseScsrmm(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                   problem.getNumOfSamples(),1,problem.getNumOfFeatures(),
                   nnz,devAlpha,descr,devValA,devRowPtrA,devColIndA,
                   devBDense,problem.getNumOfFeatures(),0,
                   pfHessianRow,problem.getNumOfSamples());
    const float_point devBSelfDot = devValASelfDot[nPosofRowAtHessian];
    RBFKernel<<<Ceil(problem.getNumOfSamples(),BLOCK_SIZE),BLOCK_SIZE>>>
            (devValASelfDot,devBSelfDot,pfHessianRow,problem.getNumOfSamples(),gamma);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(devBDense));
}

bool HostHessianOnFly::PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                         vector<vector<float_point> > &v_v_DocVector) {
    return true;
}

bool HostHessianOnFly::GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples,
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

