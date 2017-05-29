//
// Created by ss on 16-11-15.
//

#include <sys/time.h>
#include "deviceHessianOnFly.h"
#include "../constant.h"
#include "../../SharedUtility/CudaMacro.h"

__global__ void RBFKernel(const real *aSelfDot, real bSelfDot, real *dotProduct, int numOfSamples,
                          float gamma) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numOfSamples) {
        dotProduct[idx] = expf(-(aSelfDot[idx] + bSelfDot - dotProduct[idx] * 2) * gamma);
    }
}

void DeviceHessianOnFly::ReadRow(int nPosofRowAtHessian, real *devHessianRow, int start, int end) {
    const int numOfSamples = end - start;
    const int *csrRowPtr = csrMat.getCSRRowPtr();
    const int numOfFeatures = csrMat.getNumOfFeatures();
    const int nnzA = csrRowPtr[end] - csrRowPtr[start];
    const int *devARowPtr;
    if (numOfSamples == csrMat.getNumOfSamples()) {
        //for binary case
        devARowPtr = devRowPtr;
    } else {
        //for multi-class case
        devARowPtr = devRowPtrSplit + start;
        if (start != 0)
            devARowPtr++;
    }
    const real *devAVal = devVal + csrRowPtr[start];
    const int *devAColInd = devColInd + csrRowPtr[start];
    const int nnzB = csrRowPtr[nPosofRowAtHessian + 1] - csrRowPtr[nPosofRowAtHessian];
    const real *devBVal = devVal + csrRowPtr[nPosofRowAtHessian];
    const int *devBColInd = devColInd + csrRowPtr[nPosofRowAtHessian];
//    float_point *devDenseVector;
//    checkCudaErrors(cudaMalloc((void **) &devDenseVector, sizeof(float_point) * numOfFeatures));
    checkCudaErrors(cudaMemset(devDenseVector, 0, sizeof(real) * numOfFeatures));
    cusparseSsctr(handle, nnzB, devBVal, devBColInd, devDenseVector, CUSPARSE_INDEX_BASE_ZERO);
//    checkCudaErrors(cudaMemset(devHessianRow,0,sizeof(float_point) * numOfSamples));
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   numOfSamples, 1, numOfFeatures,
                   nnzA, &one, descr, devAVal, devARowPtr, devAColInd,
                   devDenseVector, numOfFeatures, &zero,
                   devHessianRow, numOfSamples);
    RBFKernel << < Ceil(numOfSamples, BLOCK_SIZE), BLOCK_SIZE >> >
                                                   (devValSelfDot +
                                                    start, csrMat.csrValSelfDot[nPosofRowAtHessian], devHessianRow, numOfSamples, gamma);
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
//    checkCudaErrors(cudaFree(devDenseVector));
}

bool DeviceHessianOnFly::PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                           vector<vector<real> > &v_v_DocVector) {
    return true;
}

bool DeviceHessianOnFly::GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples,
                                        real *pfHessianDiag) {
    for (int i = 0; i < nNumofTraingSamples; ++i) {
        pfHessianDiag[i] = 1;
    }
    return true;
}

DeviceHessianOnFly::DeviceHessianOnFly(const SvmProblem &subProblem, real gamma) :
        gamma(gamma), zero(0.0f), one(1.0f),
        csrMat(subProblem.v_vSamples, subProblem.getNumOfFeatures()) {
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    checkCudaErrors(cudaMalloc((void **) &devVal, sizeof(real) * csrMat.getNnz()));
    checkCudaErrors(cudaMalloc((void **) &devValSelfDot, sizeof(real) * csrMat.getNumOfSamples()));
    checkCudaErrors(cudaMalloc((void **) &devRowPtr, sizeof(int) * (csrMat.getNumOfSamples() + 1)));
    checkCudaErrors(cudaMalloc((void **) &devRowPtrSplit, sizeof(int) * (csrMat.getNumOfSamples() + 2)));
    checkCudaErrors(cudaMalloc((void **) &devColInd, sizeof(int) * (csrMat.getNnz())));
    checkCudaErrors(cudaMemcpy(devVal, csrMat.getCSRVal(), sizeof(real) * csrMat.getNnz(),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devValSelfDot, csrMat.getCSRValSelfDot(),
                               sizeof(real) * subProblem.v_vSamples.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devRowPtr, csrMat.getCSRRowPtr(), sizeof(int) * (subProblem.v_vSamples.size() + 1),
                               cudaMemcpyHostToDevice));
    //nnz for samples with label +1
    int nnzA = csrMat.csrRowPtr[subProblem.count[0]];
    csrRowPtrSplit = vector<int>(csrMat.csrRowPtr.begin(), csrMat.csrRowPtr.begin() + subProblem.count[0] + 1);
    for (int i = 0; i <= subProblem.count[1]; ++i) {
        csrRowPtrSplit.push_back(csrMat.csrRowPtr[subProblem.count[0] + i] - nnzA);
    }
    checkCudaErrors(cudaMemcpy(devRowPtrSplit, csrRowPtrSplit.data(), sizeof(int) * (subProblem.v_vSamples.size() + 2),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devColInd, csrMat.getCSRColInd(), sizeof(int) * (csrMat.getNnz()),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &devDenseVector, sizeof(real) * subProblem.getNumOfFeatures()));

}

DeviceHessianOnFly::~DeviceHessianOnFly() {
    checkCudaErrors(cudaFree(devVal));
    checkCudaErrors(cudaFree(devValSelfDot));
    checkCudaErrors(cudaFree(devRowPtr));
    checkCudaErrors(cudaFree(devRowPtrSplit));
    checkCudaErrors(cudaFree(devColInd));
    checkCudaErrors(cudaFree(devDenseVector));
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
}

