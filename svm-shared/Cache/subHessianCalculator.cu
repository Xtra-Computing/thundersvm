/*
 * subHessianCalculater.cu
 *
 *  Created on: 10 Jan 2017
 *      Author: Zeyi Wen
 */

#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include "subHessianCalculator.h"
#include "../constant.h"
#include "../../SharedUtility/KeyValue.h"
#include "../../SharedUtility/Timer.h"

__global__ void RBFKernel(const float_point *selfDot0, const float_point *selfDot1,
                          float_point *dotProduct, int n, int m,
                          float gamma) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = idx / m;
    int j = idx % m;
    if (idx < n * m) {
        dotProduct[idx] = expf(-(selfDot0[i] + selfDot1[j] - dotProduct[idx] * 2) * gamma);
    }
}

/**
 * @brief: create handle and descr for CSR matrix operations
 */
void SubHessianCalculator::prepareCSRContext(cusparseHandle_t &handle, cusparseMatDescr_t &descr){
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
}

/**
 * @brief: release handle and descr
 */
void SubHessianCalculator::releaseCSRContext(cusparseHandle_t &handle, cusparseMatDescr_t &descr){
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

/**
 * @brief: compute a sub/whole kernel matrix
 * @param: n is the number of rows of matrix0; m is the number of rows of matrix1; k is the dimension.
 */
void SubHessianCalculator::computeSubHessianMatrix(cusparseHandle_t handle, cusparseMatDescr_t descr,
									   CSRMatrix &csrMatrix0, int n, CSRMatrix &csrMatrix1, int m, int k,
									   float_point *devC, const SVMParam &param){
	float_point *devVal0;
	int *devRowPtr0, *devColInd0;
	csrMatrix0.copy2Dev(devVal0, devRowPtr0, devColInd0);
	float_point *devSelfDot0;
	int nnz0 = csrMatrix0.getNnz();
	checkCudaErrors(cudaMalloc((void **) &devSelfDot0, sizeof(float_point) * n));
	checkCudaErrors(cudaMemcpy(devSelfDot0, csrMatrix0.getCSRValSelfDot(), sizeof(float_point) * n, cudaMemcpyHostToDevice));

	//initialize parameters of matrix1
	int nnz1 = nnz0;
	float_point *devVal1 = devVal0;
	int *devRowPtr1 = devRowPtr0, *devColInd1 = devColInd0;
	float_point *devSelfDot1 = devSelfDot0;
	if(&csrMatrix1 != &csrMatrix0){//compare two addresses
		csrMatrix1.copy2Dev(devVal1, devRowPtr1, devColInd1);
		nnz1 = csrMatrix1.getNnz();
		checkCudaErrors(cudaMalloc((void **) &devSelfDot1, sizeof(float_point) * m));
		checkCudaErrors(cudaMemcpy(devSelfDot1, csrMatrix1.getCSRValSelfDot(), sizeof(float_point) * m, cudaMemcpyHostToDevice));
	}
	CSRMatrix::CSRmm2Dense(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, n, m, k, descr,
	                       nnz0, devVal0, devRowPtr0, devColInd0, descr, nnz1, devVal1, devRowPtr1, devColInd1, devC);
	RBFKernel << < Ceil(n * m, BLOCK_SIZE), BLOCK_SIZE >> > (devSelfDot0, devSelfDot1, devC, n, m, param.gamma);

	checkCudaErrors(cudaFree(devSelfDot0));
    csrMatrix0.freeDev(devVal0, devRowPtr0, devColInd0);
    if(&csrMatrix1 != &csrMatrix0){
    	checkCudaErrors(cudaFree(devSelfDot1));
    	csrMatrix1.freeDev(devVal1, devRowPtr1, devColInd1);
    }
}

void SubHessianCalculator::preComputeSharedCache(vector<float_point*> &hostSharedCache, const SvmProblem &problem,
                                                 const SVMParam &param) {
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);

    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        printf("pre-compute shared cache %d\n", i);
        vector<vector<KeyValue> > oneClass = problem.getOneClassSamples(i);
        int n = oneClass.size();
        int k = problem.getNumOfFeatures();
        CSRMatrix csrMatrix(oneClass, k);
        float_point *devC;
        checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * n * n));//this can be moved out of for-loop by reusing the memory.
        computeSubHessianMatrix(handle, descr, csrMatrix, n, csrMatrix, n, k, devC, param);

        checkCudaErrors(cudaMemcpy(hostSharedCache[i], devC, sizeof(float_point) * n * n, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(devC));
    }

    releaseCSRContext(handle, descr);
}

void SubHessianCalculator::preComputeUniqueCache(int i, int j, const SvmProblem &subProblem,
		    	vector<float_point*> &devUniqueCache, vector<size_t> &sizeOfEachRowInUniqueCache,
				vector<int> &numOfElementEachRowInUniqueCache, const SVMParam &param) {
    printf("pre-compute unique cache....");
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);

    int n = subProblem.count[0];
    int m = subProblem.count[1];
    int k = subProblem.getNumOfFeatures();
    vector<vector<KeyValue> > samples0(subProblem.v_vSamples.begin(), subProblem.v_vSamples.begin() + n);
    vector<vector<KeyValue> > samples1(subProblem.v_vSamples.begin() + n, subProblem.v_vSamples.begin() + n + m);
    CSRMatrix csrMatrix0(samples0, k);
    CSRMatrix csrMatrix1(samples1, k);
    float_point *devC;
    checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * n * m));
    computeSubHessianMatrix(handle, descr, csrMatrix0, n, csrMatrix1, m, k, devC, param);

    checkCudaErrors(cudaMemcpy2D(devUniqueCache[0], sizeOfEachRowInUniqueCache[0], devC,
                                 m * sizeof(float_point), m * sizeof(float_point), n, cudaMemcpyDeviceToDevice));

    //compute another sub kernel matrix by transposition
    float const alpha(1.0);
    float const beta(0.0);
    cublasHandle_t handle2;
    cublasCreate(&handle2);
    cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, devC, m, &beta, devC, n, devUniqueCache[1],
                numOfElementEachRowInUniqueCache[1]);
    cublasDestroy(handle2);

    checkCudaErrors(cudaFree(devC));
    releaseCSRContext(handle, descr);
    printf("done\n");
}

void SubHessianCalculator::preComputeAndStoreInHost(float_point *hostHessianMatrix, const SvmProblem &problem,
													bool &preComputeInHost, const SVMParam &param) {
    printf("pre-compute in host\n");
    preComputeInHost = true;
    timeval start, end;
    gettimeofday(&start,NULL);
    vector<vector<KeyValue> > permutedSamples;
    for (int i = 0; i < problem.v_vSamples.size(); ++i) {
        permutedSamples.push_back(problem.v_vSamples[problem.perm[i]]);
    }
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);

    int m = problem.getNumOfSamples();
    int k = problem.getNumOfFeatures();
    int n = m / 100;
    float_point *devValA, *devValB, *devSelfDot;
    int *devRowPtrA, *devColIndA, *devRowPtrB, *devColIndB;
    float_point *devC;
    CSRMatrix all(permutedSamples, k);
    int nnzA = all.getNnz();
    all.copy2Dev(devValA, devRowPtrA, devColIndA);
    checkCudaErrors(cudaMalloc((void **) &devSelfDot, sizeof(float_point) * m));
    checkCudaErrors(cudaMemcpy(devSelfDot, all.getCSRValSelfDot(), sizeof(float_point) * m, cudaMemcpyHostToDevice));
    printf("n = %d\n", n);
    float totalTime = 0;
    for (int i = 0; i < m / n + 1; ++i) {
        CSRMatrix sub(
                vector<vector<KeyValue> >(permutedSamples.begin() + n * i, permutedSamples.begin() + (n * (i + 1)>m?m:(n*(i+1)))),
                k);
        int tn = sub.getNumOfSamples();
        int nnzB = sub.getNnz();
        sub.copy2Dev(devValB, devRowPtrB, devColIndB);
        checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * tn * m));
        CSRMatrix::CSRmm2Dense(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, tn, m, k,
                               descr, nnzB, devValB, devRowPtrB, devColIndB, descr, nnzA, devValA, devRowPtrA,
                               devColIndA, devC);
        RBFKernel << < Ceil(tn * m, BLOCK_SIZE), BLOCK_SIZE >> >
                                                (devSelfDot + n * i, devSelfDot, devC, tn, m, param.gamma);
        sub.freeDev(devValB, devRowPtrB, devColIndB);
        checkCudaErrors(
                cudaMemcpy(hostHessianMatrix + n * m * i, devC, sizeof(float_point) * tn * m, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(devC));
    }
    checkCudaErrors(cudaFree(devSelfDot));
    releaseCSRContext(handle, descr);
    gettimeofday(&end,NULL);
    printf("time elapsed for pre-compute hessian matrix in host: %f\n", timeElapse(start,end));
}

void SubHessianCalculator::preComputeCache4BinaryProblem(float_point *devC, const SvmProblem &problem,
                                                         const SVMParam &param) {
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);
    CSRMatrix csrMatrix(problem.v_vSamples, problem.getNumOfFeatures());
    int n = problem.getNumOfSamples();
    int k = problem.getNumOfFeatures();
    computeSubHessianMatrix(handle, descr, csrMatrix, n, csrMatrix, n, k, devC, param);
    releaseCSRContext(handle, descr);
}
