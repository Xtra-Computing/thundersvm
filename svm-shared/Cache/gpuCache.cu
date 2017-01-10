//
// Created by shijiashuai on 2016/12/16.
//

#include <cublas_v2.h>
#include "gpuCache.h"
#include "../constant.h"
#include "subHessianCalculater.h"


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
    	SubHessianCalculater::preComputeUniqueCache(i, j, subProblem,
    			devUniqueCache, sizeOfEachRowInUniqueCache, numOfElementEachRowInUniqueCache, param);
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
    //SubHessianCalculater::preComputeAndStoreInHost(hostHessianMatrix, problem, preComputeInHost, param);
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
        SubHessianCalculater::preComputeSharedCache(hostSharedCache, problem, param);
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
