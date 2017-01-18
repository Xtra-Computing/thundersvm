/*
 * @author: shijiashuai
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "csrMatrix.h"

/**
 * @brief: CSR matrix constructor; construct from libsvm format data.
 */
CSRMatrix::CSRMatrix(const vector<vector<svm_node> > &samples, int numOfFeatures) : samples(samples),
                                                                                    numOfFeatures(numOfFeatures) {
    int start = 0;
    for (int i = 0; i < samples.size(); ++i) {
        csrRowPtr.push_back(start);
        int size = samples[i].size() - 1; //ignore end node for libsvm data format
        start += size;
        float_point sum = 0;
        for (int j = 0; j < size; ++j) {
            csrVal.push_back(samples[i][j].value);
            sum += samples[i][j].value * samples[i][j].value;
            csrColInd.push_back(samples[i][j].index - 1);//libsvm data format is one-based, convert it to zero-based
        }
        csrValSelfDot.push_back(sum);
    }
    csrRowPtr.push_back(start);
}

/**
 * @brief: get the number of nonzero elements of the CSR matrix.
 */
int CSRMatrix::getNnz() const {
    return csrVal.size();
}

const float_point *CSRMatrix::getCSRVal() const {
    return csrVal.data();
}

const float_point *CSRMatrix::getCSRValSelfDot() const {
    return csrValSelfDot.data();
}

const int *CSRMatrix::getCSRRowPtr() const {
    return csrRowPtr.data();
}

const int *CSRMatrix::getCSRColInd() const {
    return csrColInd.data();
}

int CSRMatrix::getNumOfSamples() const {
    return samples.size();
}

int CSRMatrix::getNumOfFeatures() const {
    return numOfFeatures;
}

/**
 * @brief: multiple two sparse matrices and output a dense matrixC.
 * @k: the dimension of training data.
 */
void CSRMatrix::CSRmm2Dense(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n,
                       int k, const cusparseMatDescr_t descrA, const int nnzA, const float *valA, const int *rowPtrA,
                       const int *colIndA, const cusparseMatDescr_t descrB, const int nnzB, const float *valB,
                       const int *rowPtrB, const int *colIndB, float *matrixC) {
    /*
     * The CSRmm2Dense result is column-major instead of row-major. To avoid transposing the result
     * we compute B'A' instead of AB' : (AB)' = B'A'
     * */
    if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
        transA = CUSPARSE_OPERATION_TRANSPOSE;
    else transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (transB == CUSPARSE_OPERATION_NON_TRANSPOSE)
        transB = CUSPARSE_OPERATION_TRANSPOSE;
    else transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float_point *devA;
    checkCudaErrors(cudaMalloc((void**)&devA,sizeof(float_point)*m*k));
    cusparseScsr2dense(handle,m,k,descrA,valA,rowPtrA,colIndA,devA,m);
    float one(1);
    float zero(0);
    cusparseScsrmm2(handle,transB,transA,n,m,k,nnzB,&one,descrB,valB,rowPtrB,colIndB,devA,m,&zero,matrixC,n);
    checkCudaErrors(cudaFree(devA));
/**
 * the code below is csr * csr, much slower than the code above.
 */
//    if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE)
//        transA = CUSPARSE_OPERATION_TRANSPOSE;
//    else transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    if (transB == CUSPARSE_OPERATION_NON_TRANSPOSE)
//        transB = CUSPARSE_OPERATION_TRANSPOSE;
//    else transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
//    cusparseMatDescr_t descrC = descrA;
//    int baseC, nnzC; // nnzTotalDevHostPtr points to host memory
//    int *nnzTotalDevHostPtr = &nnzC;
//    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
//    int *colIndC;
//    float *valC;
//    int *rowPtrC;
//    checkCudaErrors(cudaMalloc((void **) &rowPtrC, sizeof(int) * (n + 1)));
//    cusparseXcsrgemmNnz(handle, transB, transA, n, m, k, descrB, nnzB, rowPtrB, colIndB, descrA, nnzA, rowPtrA,
//                        colIndA, descrC, rowPtrC, nnzTotalDevHostPtr);
//    if (NULL != nnzTotalDevHostPtr) { nnzC = *nnzTotalDevHostPtr; }
//    else {
//        checkCudaErrors(cudaMemcpy(&nnzC, rowPtrC + m, sizeof(int), cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(&baseC, rowPtrC, sizeof(int), cudaMemcpyDeviceToHost));
//        nnzC -= baseC;
//    }
//    checkCudaErrors(cudaMalloc((void **) &colIndC, sizeof(int) * nnzC));
//    checkCudaErrors(cudaMalloc((void **) &valC, sizeof(float) * nnzC));
//    cusparseScsrgemm(handle, transB, transA, n, m, k, descrB, nnzB, valB, rowPtrB, colIndB, descrA, nnzA,
//                     valA, rowPtrA, colIndA, descrC, valC, rowPtrC, colIndC);
//    cusparseScsr2dense(handle, n, m, descrC, valC, rowPtrC, colIndC, matrixC, n);
//    checkCudaErrors(cudaFree(colIndC));
//    checkCudaErrors(cudaFree(valC));
//    checkCudaErrors(cudaFree(rowPtrC));
}

/**
 * @brief: copy the CSR matrix to device memory.
 */
void CSRMatrix::copy2Dev(float_point *&devVal, int *&devRowPtr, int *&devColInd) {

    int nnz = this->getNnz();
    checkCudaErrors(cudaMalloc((void **) &devVal, sizeof(float_point) * nnz));
    checkCudaErrors(cudaMalloc((void **) &devRowPtr, sizeof(int) * (this->getNumOfSamples() + 1)));
    checkCudaErrors(cudaMalloc((void **) &devColInd, sizeof(int) * nnz));
    checkCudaErrors(cudaMemcpy(devVal, this->getCSRVal(), sizeof(float_point) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devRowPtr, this->getCSRRowPtr(), sizeof(int) * (this->getNumOfSamples() + 1),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devColInd, this->getCSRColInd(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
}

/**
 * @brief: release the device CSR matrix
 */
void CSRMatrix::freeDev(float_point *&devVal, int *&devRowPtr, int *&devColInd) {
    checkCudaErrors(cudaFree(devVal));
    checkCudaErrors(cudaFree(devRowPtr));
    checkCudaErrors(cudaFree(devColInd));
}

