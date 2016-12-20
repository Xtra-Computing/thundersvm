//
// Created by shijiashuai on 2016/11/1.
//

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "svmProblem.h"

void SvmProblem::groupClasses() {
    if (!subProblem) {
        vector<int> dataLabel(v_nLabels.size());
        for (int i = 0; i < v_nLabels.size(); ++i) {
            int j;
            for (j = 0; j < label.size(); ++j) {
                if (v_nLabels[i] == label[j]) {
                    count[j]++;
                    break;
                }
            }
            dataLabel[i] = j;
            //if the label is unseen, add it to label set
            if (j == label.size()) {
                label.push_back(v_nLabels[i]);
                count.push_back(1);
            }
        }

        start.push_back(0);
        for (int i = 1; i < count.size(); ++i) {
            start.push_back(start[i - 1] + count[i - 1]);
        }
        vector<int> _start(start);
        perm = vector<int>(v_nLabels.size());
        for (int i = 0; i < v_nLabels.size(); ++i) {
            perm[_start[dataLabel[i]]] = i;
            _start[dataLabel[i]]++;
        }
    }
}

SvmProblem SvmProblem::getSubProblem(int i, int j) const {
    vector<vector<svm_node> > v_vSamples;
    vector<int> v_nLabels;
    vector<int> originalIndex;
    vector<int> originalLabel;
    int si = start[i];
    int ci = count[i];
    int sj = start[j];
    int cj = count[j];
    for (int k = 0; k < ci; ++k) {
        v_vSamples.push_back(this->v_vSamples[perm[si + k]]);
        originalIndex.push_back(perm[si + k]);
        originalLabel.push_back(i);
        v_nLabels.push_back(+1);
    }
    for (int k = 0; k < cj; ++k) {
        v_vSamples.push_back(this->v_vSamples[perm[sj + k]]);
        originalIndex.push_back(perm[sj + k]);
        originalLabel.push_back(j);
        v_nLabels.push_back(-1);
    }
    SvmProblem subProblem(v_vSamples, numOfFeatures, v_nLabels);
    subProblem.label.push_back(i);
    subProblem.label.push_back(j);
    subProblem.start.push_back(0);
    subProblem.start.push_back(count[i]);
    subProblem.count.push_back(count[i]);
    subProblem.count.push_back(count[j]);
    subProblem.originalIndex = originalIndex;
    subProblem.originalLabel = originalLabel;
    subProblem.subProblem = true;
    return subProblem;
}

unsigned int SvmProblem::getNumOfClasses() const {
    return (unsigned int) label.size();
}

unsigned long long SvmProblem::getNumOfSamples() const {
    return v_vSamples.size();
}

int SvmProblem::getNumOfFeatures() const {
    return numOfFeatures;
}

vector<vector<svm_node> > SvmProblem::getOneClassSamples(int i) const {
    vector<vector<svm_node> >samples;
    int si = start[i];
    int ci = count[i];
    for (int k = 0; k < ci; ++k) {
        samples.push_back(v_vSamples[perm[si+k]]);
    }
    return samples;
}

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

void
CSRMatrix::CSRmm2Dense(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n,
                       int k, const cusparseMatDescr_t descrA, const int nnzA, const float *valA, const int *rowPtrA,
                       const int *colIndA, const cusparseMatDescr_t descrB, const int nnzB, const float *valB,
                       const int *rowPtrB, const int *colIndB, float *C) {
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
    cusparseMatDescr_t descrC = descrA;
    int baseC, nnzC; // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    int *colIndC;
    float *valC;
    int *rowPtrC;
    checkCudaErrors(cudaMalloc((void **) &rowPtrC, sizeof(int) * (n + 1)));
    cusparseXcsrgemmNnz(handle, transB, transA, n, m, k, descrB, nnzB, rowPtrB, colIndB, descrA, nnzA, rowPtrA,
                        colIndA, descrC, rowPtrC, nnzTotalDevHostPtr);
    if (NULL != nnzTotalDevHostPtr) { nnzC = *nnzTotalDevHostPtr; }
    else {
        checkCudaErrors(cudaMemcpy(&nnzC, rowPtrC + m, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&baseC, rowPtrC, sizeof(int), cudaMemcpyDeviceToHost));
        nnzC -= baseC;
    }
    checkCudaErrors(cudaMalloc((void **) &colIndC, sizeof(int) * nnzC));
    checkCudaErrors(cudaMalloc((void **) &valC, sizeof(float) * nnzC));
    cusparseScsrgemm(handle, transB, transA, n, m, k, descrB, nnzB, valB, rowPtrB, colIndB, descrA, nnzA,
                     valA, rowPtrA, colIndA, descrC, valC, rowPtrC, colIndC);
    cusparseScsr2dense(handle, n, m, descrC, valC, rowPtrC, colIndC, C, n);
    checkCudaErrors(cudaFree(colIndC));
    checkCudaErrors(cudaFree(valC));
    checkCudaErrors(cudaFree(rowPtrC));
}

int CSRMatrix::getNumOfFeatures() const {
    return numOfFeatures;
}

