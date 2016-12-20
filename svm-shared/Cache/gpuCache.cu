//
// Created by shijiashuai on 2016/12/16.
//

#include "gpuCache.h"
#include "../constant.h"


void GpuCache::enable(int i, int j, const SvmProblem &subProblem) {
    //enable shared cache for class i and j
    this->subProblem = &subProblem;
    hessianCalculator = new DeviceHessianOnFly(subProblem, param.gamma);
    checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[i]),
                                    &sizeOfEachRowInCache[i], problem.count[i] * sizeof(float_point), cacheSize[i]));
    checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[j]),
                                    &sizeOfEachRowInCache[j], problem.count[j] * sizeof(float_point), cacheSize[j]));
    numOfElementEachRowInCache[i] = sizeOfEachRowInCache[i] / sizeof(float_point);
    numOfElementEachRowInCache[j] = sizeOfEachRowInCache[j] / sizeof(float_point);
    int uniqueCacheRowLength = max(problem.count[i], problem.count[j]);
    int uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 4 / uniqueCacheRowLength, cacheSize[i] + cacheSize[j]);
    printf("unique cache row length %d, unique cache size %d\n", uniqueCacheRowLength, uniqueCacheSize);
    checkCudaErrors(cudaMallocPitch((void **) &devUniqueCache,
                                    &sizeOfEachRowInUniqueCache,
                                    uniqueCacheRowLength * sizeof(float_point),
                                    uniqueCacheSize));
    numOfElementEachRowInUniqueCache = sizeOfEachRowInUniqueCache / sizeof(float_point);
    uniqueCacheStrategy = new CLATCache(problem.count[i]+ problem.count[j]);
    uniqueCacheStrategy->SetCacheSize(uniqueCacheSize);
    uniqueCacheStrategy->InitializeCache(uniqueCacheSize,problem.count[i] + problem.count[j]);
    checkCudaErrors(cudaMemcpy2D(
            devSharedCache[i], sizeOfEachRowInCache[i],
            hostSharedCache[i], problem.count[i] * sizeof(float_point),
            problem.count[i] * sizeof(float_point), cacheSize[i], cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(
            devSharedCache[j], sizeOfEachRowInCache[j],
            hostSharedCache[j], problem.count[j] * sizeof(float_point),
            problem.count[j] * sizeof(float_point), cacheSize[j], cudaMemcpyHostToDevice));
}

void GpuCache::disable(int i, int j) {
    delete hessianCalculator;
    delete uniqueCacheStrategy;
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
    checkCudaErrors(cudaFree(devUniqueCache));
}

GpuCache::GpuCache(const SvmProblem &problem, const SVMParam &param) :
        problem(problem), param(param),
        numOfElementEachRowInCache(problem.getNumOfClasses()),
        devSharedCache(problem.getNumOfClasses(), NULL),
        sizeOfEachRowInCache(problem.getNumOfClasses()) {
    for (int i = 0; i < problem.getNumOfClasses(); ++i) {
        int rowLength = problem.count[i];
        sharedCacheStrategy.push_back(new CLATCache(rowLength));
        cacheSize.push_back(min(CACHE_SIZE * 1024 * 1024 / 4 / rowLength / 3, rowLength));
        printf("shared cache %d size=%d, #samples in class %d=%d\n", i, cacheSize[i], i, rowLength);
        sharedCacheStrategy[i]->SetCacheSize(cacheSize[i]);
        sharedCacheStrategy[i]->InitializeCache(cacheSize[i], rowLength);
        hostSharedCache.push_back(new float_point[cacheSize[i] * rowLength]);
    }
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
    bool cacheHit = uniqueCacheStrategy->GetDataFromCache(rowIndex, cacheLocation, cacheFull);
    if (!cacheHit) {
        if (cacheFull)
            uniqueCacheStrategy->ReplaceExpired(rowIndex, cacheLocation, NULL);
//        printf("unique cache miss, save to location %d, ", cacheLocation);
        hessianCalculator->ReadRow(rowIndex,
//                                   devHessianRow+uniqueCacheStart,
                                   devUniqueCache + cacheLocation * numOfElementEachRowInUniqueCache,
                                   uniqueCacheStart,
                                   uniqueCacheStart + uniqueCacheCount);
    } else {
//        printf("unique cache hit at %d, ", cacheLocation);
    };
    checkCudaErrors(cudaMemcpy(
            devHessianRow + uniqueCacheStart,
            devUniqueCache + cacheLocation * numOfElementEachRowInUniqueCache,
            sizeof(float_point) * uniqueCacheCount,
            cudaMemcpyDeviceToDevice));

    //query shared cache
    int sharedCacheOffset = 0;
    if (label == 1) sharedCacheOffset = -uniqueCacheCount;
//    printf("offset is %d, ", sharedCacheOffset);
    cacheHit = sharedCacheStrategy[originalLabel]->GetDataFromCache(rowIndex + sharedCacheOffset, cacheLocation,
                                                                    cacheFull);
    if (!cacheHit) {
        if (cacheFull)
            sharedCacheStrategy[originalLabel]->ReplaceExpired(rowIndex + sharedCacheOffset, cacheLocation, NULL);
//        printf("shared cache %d miss, save to location %d.\n", originalLabel, cacheLocation);
        hessianCalculator->ReadRow(rowIndex,
                                   devSharedCache[originalLabel] +
                                   cacheLocation * numOfElementEachRowInCache[originalLabel],
//                                   devHessianRow+sharedCacheStart,
                                   sharedCacheStart,
                                   sharedCacheStart + sharedCacheCount);
    } else {
//        printf("shared cache %d hit at %d.\n", originalLabel, cacheLocation);
    }
    checkCudaErrors(cudaMemcpy(
            devHessianRow + sharedCacheStart,
            devSharedCache[originalLabel] + cacheLocation * numOfElementEachRowInCache[originalLabel],
            sizeof(float_point) * sharedCacheCount,
            cudaMemcpyDeviceToDevice));
}
