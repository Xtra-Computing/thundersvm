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

void DeviceHessianOnFly::ReadRow(int nPosofRowAtHessian, float_point *devHessianRow) {
    const int numOfSamples = csrMat.getNumOfSamples();
    const int numOfFeatures = csrMat.getNumOfFeatures();
    const int *csrRowPtr = csrMat.getCSRRowPtr();
    const int nnzB = csrRowPtr[nPosofRowAtHessian + 1] - csrRowPtr[nPosofRowAtHessian];
    const float_point *devBVal = devValA + csrRowPtr[nPosofRowAtHessian];
    const int *devBInd = devColIndA + csrRowPtr[nPosofRowAtHessian];
    float_point *devBDense;
    checkCudaErrors(cudaMalloc((void **) &devBDense, sizeof(float_point) * numOfFeatures));
    checkCudaErrors(cudaMemset(devBDense, 0, sizeof(float_point) * numOfFeatures));
    cusparseSsctr(handle, nnzB, devBVal, devBInd, devBDense, CUSPARSE_INDEX_BASE_ZERO);
//    float_point *devC;
//    checkCudaErrors(cudaMalloc((void **) &devHessianRow, sizeof(float_point) * numOfSamples));
    checkCudaErrors(cudaMemset(devHessianRow,0,sizeof(float_point) * numOfSamples));
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   numOfSamples, 1, numOfFeatures,
                   nnz, &one, descr, devValA, devRowPtrA, devColIndA,
                   devBDense, numOfFeatures, &zero,
                   devHessianRow, numOfSamples);
    RBFKernel << < Ceil(numOfSamples, BLOCK_SIZE), BLOCK_SIZE >> >
                                                                (devValASelfDot, nPosofRowAtHessian, devHessianRow, numOfSamples, gamma);
//    float_point *hrow = new float_point[numOfSamples];
//    checkCudaErrors(
//            cudaMemcpy(hrow, devHessianRow, sizeof(float_point) * numOfSamples, cudaMemcpyDeviceToHost));
//    RBFKernelFunction function(gamma);
//    float_point *hostKernel = new float_point[problem.getNumOfSamples()];
//    float_point totalErr = 0;
//    vector<vector<svm_node> > s = problem.v_vSamples;
//    function.ComputeSparseRow(s,nPosofRowAtHessian,1,hostKernel);
//    for (int i = 0; i < problem.getNumOfSamples(); ++i) {
//       float_point err = fabs(hostKernel[i] - hrow[i]);
//        totalErr +=err;
//        printf("row %d, col %d, host %f, device %f,err %f\n",nPosofRowAtHessian, i, hostKernel[i],hrow[i],err);
//    }
//    printf("compute row %d, total err %f\n",nPosofRowAtHessian,totalErr);
//    memcpy(devHessianRow,hostKernel,sizeof(float_point) * numOfSamples);
//    delete[] hostKernel;
    checkCudaErrors(cudaFree(devBDense));
//    checkCudaErrors(cudaFree(devC));
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

