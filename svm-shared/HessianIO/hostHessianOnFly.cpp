//
// Created by ss on 16-11-15.
//

#include "hostHessianOnFly.h"

void HostHessianOnFly::ReadRow(int nPosofRowAtHessian, float_point *pfHessianRow) {
    kernelCalculator.ComputeSparseRow(samples,nPosofRowAtHessian,1,pfHessianRow);
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

