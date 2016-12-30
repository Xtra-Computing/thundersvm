//
// Created by shijiashuai on 2016/12/16.
//

#include <cublas_v2.h>
#include "gpuCache.h"
#include "../constant.h"


void GpuCache::enable(int i, int j, const SvmProblem &subProblem) {
    //enable shared cache for class i and j
    this->subProblem = &subProblem;
//    hessianCalculator = new DeviceHessianOnFly(subProblem, param.gamma);
    checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[i]),
                                    &sizeOfEachRowInCache[i], problem.count[i] * sizeof(float_point), cacheSize[i]));
    checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[j]),
                                    &sizeOfEachRowInCache[j], problem.count[j] * sizeof(float_point), cacheSize[j]));
    numOfElementEachRowInCache[i] = sizeOfEachRowInCache[i] / sizeof(float_point);
    numOfElementEachRowInCache[j] = sizeOfEachRowInCache[j] / sizeof(float_point);
    int uniqueCacheRowLength = problem.count[j];
    int uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 4 / uniqueCacheRowLength, cacheSize[i]);
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
    uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 4 / uniqueCacheRowLength, cacheSize[j]);
    printf("unique cache 1 row length %d, size %d\n", uniqueCacheRowLength, uniqueCacheSize);
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

    preComputeUniqueCache(i, j, subProblem);
}

void GpuCache::disable(int i, int j) {
//    delete hessianCalculator;
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
        sizeOfEachRowInUniqueCache(2) {
    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        int rowLength = problem.count[i];
        sharedCacheStrategy.push_back(new CLATCache(rowLength));
        cacheSize.push_back(min(CACHE_SIZE * 1024 * 1024 / 4 / rowLength / 3, rowLength));
        printf("shared cache %d size=%d, #samples in class %d=%d\n", i, cacheSize[i], i, rowLength);
        sharedCacheStrategy[i]->SetCacheSize(cacheSize[i]);
        sharedCacheStrategy[i]->InitializeCache(cacheSize[i], rowLength);
        hostSharedCache.push_back(new float_point[cacheSize[i] * rowLength]);
    }
    preComputeSharedCache();
}

GpuCache::~GpuCache() {
    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        delete sharedCacheStrategy[i];
        delete[] hostSharedCache[i];
    }
}

void GpuCache::getHessianRow(int rowIndex, float_point *devHessianRow) {
    int originalLabel = subProblem->originalLabel[rowIndex];

//    printf("query row %d, label %d, ", rowIndex, originalLabel);
    //map +1 -1 to 0 1
    int label = 1 - (subProblem->v_nLabels[rowIndex] + 1) / 2;
    int sharedCacheStart = subProblem->start[label];
    int uniqueCacheStart = subProblem->start[1 - label];
    int sharedCacheCount = subProblem->count[label];
    int uniqueCacheCount = subProblem->count[1 - label];
//    printf("original label %d, label %d\n",originalLabel, label);
//    printf("shared cache start %d, unique cache start %d\n",sharedCacheStart,uniqueCacheStart);
//    printf("row index %d\n",rowIndex);


    int cacheLocation;
    bool cacheFull = false;

    //query unique cache
    int uniqueCacheOffset = -subProblem->start[label];
//    bool cacheHit = uniqueCacheStrategy[label]->GetDataFromCache(rowIndex + uniqueCacheOffset, cacheLocation,
//                                                                 cacheFull);
//    if (!cacheHit) {
//        if (cacheFull)
//            uniqueCacheStrategy[label]->ReplaceExpired(rowIndex + uniqueCacheOffset, cacheLocation, NULL);
////        printf("unique cache miss, save to location %d, ", cacheLocation);
//        hessianCalculator->ReadRow(rowIndex,
////                                   devHessianRow+uniqueCacheStart,
//                                   devUniqueCache[label] + cacheLocation * numOfElementEachRowInUniqueCache[label],
//                                   uniqueCacheStart,
//                                   uniqueCacheStart + uniqueCacheCount);
//    } else {
////        printf("unique cache hit at %d, ", cacheLocation);
//    };
    cacheLocation = rowIndex + uniqueCacheOffset;
    checkCudaErrors(cudaMemcpy(
            devHessianRow + uniqueCacheStart,
            devUniqueCache[label] + cacheLocation * numOfElementEachRowInUniqueCache[label],
            sizeof(float_point) * uniqueCacheCount,
            cudaMemcpyDeviceToDevice));

    //query shared cache
    int sharedCacheOffset = -subProblem->start[label];
//    printf("offset is %d, ", sharedCacheOffset);
//    cacheHit = sharedCacheStrategy[originalLabel]->GetDataFromCache(rowIndex + sharedCacheOffset, cacheLocation,
//                                                                    cacheFull);
//    if (!cacheHit) {
//        if (cacheFull)
//            sharedCacheStrategy[originalLabel]->ReplaceExpired(rowIndex + sharedCacheOffset, cacheLocation, NULL);
////        printf("shared cache %d miss, save to location %d.\n", originalLabel, cacheLocation);
//        hessianCalculator->ReadRow(rowIndex,
//                                   devSharedCache[originalLabel] +
//                                   cacheLocation * numOfElementEachRowInCache[originalLabel],
////                                   devHessianRow+sharedCacheStart,
//                                   sharedCacheStart,
//                                   sharedCacheStart + sharedCacheCount);
//    } else {
////        printf("shared cache %d hit at %d.\n", originalLabel, cacheLocation);
//    }
    cacheLocation = rowIndex + sharedCacheOffset;
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

void GpuCache::preComputeSharedCache() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        printf("pre-compute shared cache %d\n", i);
        vector<vector<svm_node> > oneClass = problem.getOneClassSamples(i);
        int n = oneClass.size();
        int k = problem.getNumOfFeatures();
        float_point *devVal = NULL;
        int *devRowPtr = NULL, *devColInd = NULL;
        CSRMatrix csrMatrix(oneClass, k);
        int nnz = csrMatrix.getNnz();
        csrMatrix.copy2Dev(devVal, devRowPtr, devColInd);
        float_point *devC;
        checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * n * n));
        CSRMatrix::CSRmm2Dense(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, n, n,
                               k,
                               descr, nnz, devVal, devRowPtr, devColInd, descr, nnz, devVal,
                               devRowPtr,
                               devColInd, devC);
        float_point *devSelfDot;
        checkCudaErrors(cudaMalloc((void **) &devSelfDot, sizeof(float_point) * n));
        checkCudaErrors(
                cudaMemcpy(devSelfDot, csrMatrix.getCSRValSelfDot(), sizeof(float_point) * n, cudaMemcpyHostToDevice));
        RBFKernel << < Ceil(n * n, BLOCK_SIZE), BLOCK_SIZE >> > (devSelfDot, devSelfDot, devC, n, n, param.gamma);
        checkCudaErrors(cudaMemcpy(hostSharedCache[i], devC, sizeof(float_point) * n * n, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(devC));
        csrMatrix.freeDev(devVal, devRowPtr, devColInd);
    }
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

void GpuCache::preComputeUniqueCache(int i, int j, const SvmProblem &subProblem) {
    printf("pre-compute unique cache....");
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    int n = subProblem.count[0];
    int m = subProblem.count[1];
    int k = subProblem.getNumOfFeatures();
    vector<vector<svm_node> > samples0(subProblem.v_vSamples.begin(), subProblem.v_vSamples.begin() + n);
    vector<vector<svm_node> > samples1(subProblem.v_vSamples.begin() + n, subProblem.v_vSamples.begin() + n + m);
    CSRMatrix csrMatrix0(samples0, k);
    CSRMatrix csrMatrix1(samples1, k);
    float_point *devVal0, *devVal1;
    int *devRowPtr0, *devRowPtr1, *devColInd0, *devColInd1;
    csrMatrix0.copy2Dev(devVal0, devRowPtr0, devColInd0);
    csrMatrix1.copy2Dev(devVal1, devRowPtr1, devColInd1);
    int nnz0 = csrMatrix0.getNnz();
    int nnz1 = csrMatrix1.getNnz();
    float_point *devSelfDot0, *devSelfDot1;
    checkCudaErrors(cudaMalloc((void **) &devSelfDot0, sizeof(float_point) * n));
    checkCudaErrors(cudaMalloc((void **) &devSelfDot1, sizeof(float_point) * m));
    checkCudaErrors(
            cudaMemcpy(devSelfDot0, csrMatrix0.getCSRValSelfDot(), sizeof(float_point) * n, cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(devSelfDot1, csrMatrix1.getCSRValSelfDot(), sizeof(float_point) * m, cudaMemcpyHostToDevice));
    float_point *devC;
    checkCudaErrors(cudaMalloc((void **) &devC, sizeof(float_point) * n * m));
    CSRMatrix::CSRmm2Dense(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, n, m,
                           k,
                           descr, nnz0, devVal0, devRowPtr0, devColInd0, descr, nnz1, devVal1,
                           devRowPtr1,
                           devColInd1, devC);
    RBFKernel << < Ceil(n * m, BLOCK_SIZE), BLOCK_SIZE >> > (devSelfDot0, devSelfDot1, devC, n, m, param.gamma);
    checkCudaErrors(cudaMemcpy2D(devUniqueCache[0], sizeOfEachRowInUniqueCache[0], devC,
                                 m * sizeof(float_point), m * sizeof(float_point), n, cudaMemcpyDeviceToDevice));
    float const alpha(1.0);
    float const beta(0.0);
    cublasHandle_t handle2;
    cublasCreate(&handle2);
    cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, devC, m, &beta, devC, n, devUniqueCache[1], numOfElementEachRowInUniqueCache[1]);
    cublasDestroy(handle2);
    csrMatrix0.freeDev(devVal0,devRowPtr0,devColInd0);
    csrMatrix1.freeDev(devVal1,devRowPtr1,devColInd1);
    checkCudaErrors(cudaFree(devC));
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    printf("done\n");
}
