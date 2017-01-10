//
// Created by shijiashuai on 2016/12/16.
//

#include <cublas_v2.h>
#include "gpuCache.h"
#include "../constant.h"


void GpuCache::enable(int i, int j, const SvmProblem &subProblem) {
    //enable shared cache for class i and j
    this->subProblem = &subProblem;
    canPreComputeUniqueCache = false;
    checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[i]),
                                    &sizeOfEachRowInCache[i], problem.count[i] * sizeof(float_point), cacheSize[i]));
    checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[j]),
                                    &sizeOfEachRowInCache[j], problem.count[j] * sizeof(float_point), cacheSize[j]));
    numOfElementEachRowInCache[i] = sizeOfEachRowInCache[i] / sizeof(float_point);
    numOfElementEachRowInCache[j] = sizeOfEachRowInCache[j] / sizeof(float_point);
    int uniqueCacheRowLength = problem.count[j];
    int uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 6 / uniqueCacheRowLength, cacheSize[i]);
    if (cacheSize[i] < problem.count[i]) canPreComputeUniqueCache = false;
    printf("unique cache 0 row length %d, size %d\n", uniqueCacheRowLength, uniqueCacheSize);
    checkCudaErrors(cudaMallocPitch((void **) &devUniqueCache[0],
                                    &sizeOfEachRowInUniqueCache[0],
                                    uniqueCacheRowLength * sizeof(float_point),
                                    uniqueCacheSize));
    numOfElementEachRowInUniqueCache[0] = sizeOfEachRowInUniqueCache[0] / sizeof(float_point);
    uniqueCacheStrategy[0] = new CLATCache(problem.count[i]);
    uniqueCacheStrategy[0]->SetCacheSize(uniqueCacheSize);
    uniqueCacheStrategy[0]->InitializeCache(uniqueCacheSize, problem.count[i]);

    uniqueCacheRowLength = problem.count[i];
    uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 6 / uniqueCacheRowLength, cacheSize[j]);
    printf("unique cache 1 row length %d, size %d\n", uniqueCacheRowLength, uniqueCacheSize);
    if (cacheSize[j] < problem.count[j]) canPreComputeUniqueCache = false;
    checkCudaErrors(cudaMallocPitch((void **) &devUniqueCache[1],
                                    &sizeOfEachRowInUniqueCache[1],
                                    uniqueCacheRowLength * sizeof(float_point),
                                    uniqueCacheSize));
    numOfElementEachRowInUniqueCache[1] = sizeOfEachRowInUniqueCache[1] / sizeof(float_point);
    uniqueCacheStrategy[1] = new CLATCache(problem.count[j]);
    uniqueCacheStrategy[1]->SetCacheSize(uniqueCacheSize);
    uniqueCacheStrategy[1]->InitializeCache(uniqueCacheSize, problem.count[j]);
    checkCudaErrors(cudaMemcpy2D(
            devSharedCache[i], sizeOfEachRowInCache[i],
            hostSharedCache[i], problem.count[i] * sizeof(float_point),
            problem.count[i] * sizeof(float_point), cacheSize[i], cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(
            devSharedCache[j], sizeOfEachRowInCache[j],
            hostSharedCache[j], problem.count[j] * sizeof(float_point),
            problem.count[j] * sizeof(float_point), cacheSize[j], cudaMemcpyHostToDevice));

    if (canPreComputeUniqueCache) {
        preComputeUniqueCache(i, j, subProblem);
    } else {
        if (!preComputeInHost) {
            printf("compute unique kernels on fly\n");
            hessianCalculator = new DeviceHessianOnFly(subProblem, param.gamma);
        } else
            printf("use pre-compute hessian matrix in host\n");
    }
}

void GpuCache::disable(int i, int j) {
    if (!canPreComputeUniqueCache)
        delete hessianCalculator;
    delete uniqueCacheStrategy[0];
    delete uniqueCacheStrategy[1];
    checkCudaErrors(cudaMemcpy2D(
            hostSharedCache[i], problem.count[i] * sizeof(float_point),
            devSharedCache[i], sizeOfEachRowInCache[i],
            problem.count[i] * sizeof(float_point), cacheSize[i], cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(
            hostSharedCache[j], problem.count[j] * sizeof(float_point),
            devSharedCache[j], sizeOfEachRowInCache[j],
            problem.count[j] * sizeof(float_point), cacheSize[j], cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(devSharedCache[i]));
    checkCudaErrors(cudaFree(devSharedCache[j]));
    checkCudaErrors(cudaFree(devUniqueCache[0]));
    checkCudaErrors(cudaFree(devUniqueCache[1]));
}

GpuCache::GpuCache(const SvmProblem &problem, const SVMParam &param) :
        problem(problem), param(param),
        numOfElementEachRowInCache(problem.getNumOfClasses()),
        devSharedCache(problem.getNumOfClasses(), NULL),
        sizeOfEachRowInCache(problem.getNumOfClasses()),
        devUniqueCache(2),
        uniqueCacheStrategy(2),
        numOfElementEachRowInUniqueCache(2),
        sizeOfEachRowInUniqueCache(2),
        canPreComputeSharedCache(false),
        preComputeInHost(false) {
    checkCudaErrors(cudaMallocHost((void **) &hostHessianMatrix,
                                   sizeof(float_point) * problem.getNumOfSamples() * problem.getNumOfSamples()));
    //preComputeAndStoreInHost();
    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        int rowLength = problem.count[i];
        sharedCacheStrategy.push_back(new CLATCache(rowLength));
        cacheSize.push_back(min(CACHE_SIZE * 1024 * 256 / rowLength / 3, rowLength));
        printf("shared cache %d size=%d, #samples in class %d=%d\n", i, cacheSize[i], i, rowLength);
        if (cacheSize[i] < problem.count[i]) canPreComputeSharedCache = false;
        sharedCacheStrategy[i]->SetCacheSize(cacheSize[i]);
        sharedCacheStrategy[i]->InitializeCache(cacheSize[i], rowLength);
        hostSharedCache.push_back(new float_point[cacheSize[i] * rowLength]);
    }
    if (canPreComputeSharedCache) {
        printf("cache is large enough, pre-computing shared cache\n");
        preComputeSharedCache();
    } else {
        if (!preComputeInHost)
            printf("compute shared kernels on fly\n");
        else
            printf("use pre-compute hessian matrix in host\n");
    }
}

GpuCache::~GpuCache() {
    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        delete sharedCacheStrategy[i];
        delete[] hostSharedCache[i];
    }
    checkCudaErrors(cudaFreeHost(hostHessianMatrix));
}

void GpuCache::getHessianRow(int rowIndex, float_point *devHessianRow) {
    int originalLabel = subProblem->originalLabel[rowIndex]; //label in 0,1,2,3,4,...
    int originalIndex = subProblem->originalIndex[rowIndex];
    int label = 1 - (subProblem->v_nLabels[rowIndex] + 1) / 2; //map +1 -1 to 0 1
    int theOtherLabel = subProblem->label[1 - label];
    int sharedCacheStart = subProblem->start[label];
    int uniqueCacheStart = subProblem->start[1 - label];
    int sharedCacheCount = subProblem->count[label];
    int uniqueCacheCount = subProblem->count[1 - label];
    int uniqueCacheOffset = -subProblem->start[label];//TODO optimize here
    int sharedCacheOffset = -subProblem->start[label];

    int cacheLocation;
    bool cacheFull = false;
    bool cacheHit;

    //query unique cache
    if (canPreComputeUniqueCache) {
        cacheLocation = rowIndex + uniqueCacheOffset;
    } else {
        cacheHit = uniqueCacheStrategy[label]->GetDataFromCache(rowIndex + uniqueCacheOffset, cacheLocation,
                                                                cacheFull);
        if (!cacheHit) {
            if (cacheFull)
                uniqueCacheStrategy[label]->ReplaceExpired(rowIndex + uniqueCacheOffset, cacheLocation, NULL);

            //unique cache position for this row
            float_point *tempUniqueCachePos = devUniqueCache[label] +
            								  cacheLocation * numOfElementEachRowInUniqueCache[label];
            if (preComputeInHost)
                checkCudaErrors(cudaMemcpy(tempUniqueCachePos,
                                           hostHessianMatrix
                                           + problem.getNumOfSamples() *
                                             (problem.start[originalLabel] + rowIndex + sharedCacheOffset)
                                           + problem.start[theOtherLabel],
                                           uniqueCacheCount * sizeof(float_point),
                                           cudaMemcpyHostToDevice));
            else
                hessianCalculator->ReadRow(rowIndex, tempUniqueCachePos, uniqueCacheStart,
                						   uniqueCacheStart + uniqueCacheCount);
        }
    }
    checkCudaErrors(cudaMemcpy(
            devHessianRow + uniqueCacheStart,
            devUniqueCache[label] + cacheLocation * numOfElementEachRowInUniqueCache[label],
            sizeof(float_point) * uniqueCacheCount,
            cudaMemcpyDeviceToDevice));

    //query shared cache
    if (canPreComputeSharedCache) {
        cacheLocation = rowIndex + sharedCacheOffset;
    } else {
        cacheHit = sharedCacheStrategy[originalLabel]->GetDataFromCache(rowIndex + sharedCacheOffset, cacheLocation,
                                                                        cacheFull);
        if (!cacheHit) {
            if (cacheFull)
                sharedCacheStrategy[originalLabel]->ReplaceExpired(rowIndex + sharedCacheOffset, cacheLocation,
                                                                   NULL);
            //shared cache position
            float_point *tempSharedCachePos = devSharedCache[originalLabel] +
                    						  cacheLocation * numOfElementEachRowInCache[originalLabel];
            if (preComputeInHost)
                checkCudaErrors(cudaMemcpy(tempSharedCachePos,
                                           hostHessianMatrix
                                           + problem.getNumOfSamples() *
                                             (problem.start[originalLabel] + rowIndex + sharedCacheOffset)
                                           + problem.start[originalLabel],
                                           sharedCacheCount * sizeof(float_point),
                                           cudaMemcpyHostToDevice));
            else
                hessianCalculator->ReadRow(rowIndex, tempSharedCachePos, sharedCacheStart,
                                           sharedCacheStart + sharedCacheCount);
        }
    }
    checkCudaErrors(cudaMemcpy(
            devHessianRow + sharedCacheStart,
            devSharedCache[originalLabel] + cacheLocation * numOfElementEachRowInCache[originalLabel],
            sizeof(float_point) * sharedCacheCount,
            cudaMemcpyDeviceToDevice));
}

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
void GpuCache::prepareCSRContext(cusparseHandle_t &handle, cusparseMatDescr_t &descr){
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
}

/**
 * @brief: release handle and descr
 */
void GpuCache::releaseCSRContext(cusparseHandle_t &handle, cusparseMatDescr_t &descr){
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

/**
 * @brief: compute a sub/whole kernel matrix
 * @param: n is the number of rows of matrix0; m is the number of rows of matrix1; k is the dimension.
 */
void GpuCache::computeSubHessianMatrix(cusparseHandle_t handle, cusparseMatDescr_t descr,
									   CSRMatrix &csrMatrix0, int n, CSRMatrix &csrMatrix1, int m, int k,
									   float_point *devC){
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

void GpuCache::preComputeSharedCache() {
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);

    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        printf("pre-compute shared cache %d\n", i);
        vector<vector<svm_node> > oneClass = problem.getOneClassSamples(i);
        int n = oneClass.size();
        int k = problem.getNumOfFeatures();
        CSRMatrix csrMatrix(oneClass, k);
        float_point *devC;
        checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * n * n));//this can be moved out of for-loop by reusing the memory.
        computeSubHessianMatrix(handle, descr, csrMatrix, n, csrMatrix, n, k, devC);

        checkCudaErrors(cudaMemcpy(hostSharedCache[i], devC, sizeof(float_point) * n * n, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(devC));
    }

    releaseCSRContext(handle, descr);
}

void GpuCache::preComputeUniqueCache(int i, int j, const SvmProblem &subProblem) {
    printf("pre-compute unique cache....");
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);

    int n = subProblem.count[0];
    int m = subProblem.count[1];
    int k = subProblem.getNumOfFeatures();
    vector<vector<svm_node> > samples0(subProblem.v_vSamples.begin(), subProblem.v_vSamples.begin() + n);
    vector<vector<svm_node> > samples1(subProblem.v_vSamples.begin() + n, subProblem.v_vSamples.begin() + n + m);
    CSRMatrix csrMatrix0(samples0, k);
    CSRMatrix csrMatrix1(samples1, k);
    float_point *devC;
    checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * n * m));
    computeSubHessianMatrix(handle, descr, csrMatrix0, n, csrMatrix1, m, k, devC);

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

void GpuCache::preComputeAndStoreInHost() {
    printf("pre-compute in host\n");
    preComputeInHost = true;
    clock_t start, end;
    start = clock();
    vector<vector<svm_node> > permutedSamples;
    for (int i = 0; i < problem.v_vSamples.size(); ++i) {
        permutedSamples.push_back(problem.v_vSamples[problem.perm[i]]);
    }
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    prepareCSRContext(handle, descr);

    int m = problem.getNumOfSamples();
    int k = problem.getNumOfFeatures();
    int n = m / 20;
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
                vector<vector<svm_node> >(permutedSamples.begin() + n * i, permutedSamples.begin() + (n * (i + 1)>m?m:(n*(i+1)))),
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
        totalTime += (float) (end - start) / CLOCKS_PER_SEC;
        sub.freeDev(devValB, devRowPtrB, devColIndB);
        checkCudaErrors(
                cudaMemcpy(hostHessianMatrix + n * m * i, devC, sizeof(float_point) * tn * m, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(devC));
    }
    checkCudaErrors(cudaFree(devSelfDot));
    releaseCSRContext(handle, descr);
    end = clock();
    printf("time elapsed for pre-compute hessian matrix in host: %f\n", (float) (end - start) / CLOCKS_PER_SEC);
}
