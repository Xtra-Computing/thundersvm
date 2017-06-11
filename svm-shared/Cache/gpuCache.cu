//
// Created by shijiashuai on 2016/12/16.
//

#include <cublas_v2.h>
#include <sys/time.h>
#include "gpuCache.h"
#include "../constant.h"
#include "subHessianCalculator.h"
#include "../../SharedUtility/Timer.h"

void GpuCache::enable(int i, int j, const SvmProblem &subProblem) {
    if (binary) {
        //for binary case, use 1 shared cache to store the whole rows of hessian matrix
        checkCudaErrors(cudaMallocPitch((void **) &devSharedCache[0],
                                        &sizeOfEachRowInCache[0],
                                        problem.getNumOfSamples() * sizeof(real),
                                        cacheSize[0]));
        numOfElementEachRowInCache[0] = sizeOfEachRowInCache[0] / sizeof(real);
        if (canPreComputeSharedCache) {
            printf("cache is large enough, pre-computing\n");
            real *devC;
            checkCudaErrors(cudaMalloc((void **) &devC,
                                       sizeof(real) * problem.getNumOfSamples() * problem.getNumOfSamples()));
            //sub-problem is the same as problem but permuted
            ACCUMULATE_TIME(preComputeTimer,
                            SubHessianCalculator::preComputeCache4BinaryProblem(devC, subProblem, param);
            )
            checkCudaErrors(cudaMemcpy2D(devSharedCache[0],
                                         sizeOfEachRowInCache[0],
                                         devC,
                                         sizeof(real) * problem.getNumOfSamples(),
                                         sizeof(real) * problem.getNumOfSamples(),
                                         cacheSize[0],
                                         cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaFree(devC));
        } else {
            hessianCalculator = new DeviceHessianOnFly(subProblem, param.gamma);
        }
    } else {
        //enable shared cache for class i and j
        this->subProblem = &subProblem;
        canPreComputeUniqueCache = true;

        //allocate memory for two shared caches
        checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[i]),
                                        &sizeOfEachRowInCache[i], problem.count[i] * sizeof(real),
                                        cacheSize[i]));
        checkCudaErrors(cudaMallocPitch((void **) &(devSharedCache[j]),
                                        &sizeOfEachRowInCache[j], problem.count[j] * sizeof(real),
                                        cacheSize[j]));
        numOfElementEachRowInCache[i] = sizeOfEachRowInCache[i] / sizeof(real);
        numOfElementEachRowInCache[j] = sizeOfEachRowInCache[j] / sizeof(real);

        //allocate memory for the first unique cache
        int uniqueCacheRowLength = problem.count[j];
        int uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 6 / uniqueCacheRowLength, cacheSize[i]);
        if (cacheSize[i] < problem.count[i]) canPreComputeUniqueCache = false;
        printf("unique cache 0 row length %d, size %d\n", uniqueCacheRowLength, uniqueCacheSize);

        checkCudaErrors(cudaMallocPitch((void **) &devUniqueCache[0],
                                        &sizeOfEachRowInUniqueCache[0],
                                        uniqueCacheRowLength * sizeof(real),
                                        uniqueCacheSize));
        numOfElementEachRowInUniqueCache[0] = sizeOfEachRowInUniqueCache[0] / sizeof(real);
        uniqueCacheStrategy[0] = new CLATCache(problem.count[i]);
        uniqueCacheStrategy[0]->SetCacheSize(uniqueCacheSize);
        uniqueCacheStrategy[0]->InitializeCache(uniqueCacheSize, problem.count[i]);
        //allocate memory for the second unique cache
        uniqueCacheRowLength = problem.count[i];
        uniqueCacheSize = min(CACHE_SIZE * 1024 * 1024 / 6 / uniqueCacheRowLength, cacheSize[j]);
        printf("unique cache 1 row length %d, size %d\n", uniqueCacheRowLength, uniqueCacheSize);
        if (cacheSize[j] < problem.count[j]) canPreComputeUniqueCache = false;
        checkCudaErrors(cudaMallocPitch((void **) &devUniqueCache[1],
                                        &sizeOfEachRowInUniqueCache[1],
                                        uniqueCacheRowLength * sizeof(real),
                                        uniqueCacheSize));
        numOfElementEachRowInUniqueCache[1] = sizeOfEachRowInUniqueCache[1] / sizeof(real);
        uniqueCacheStrategy[1] = new CLATCache(problem.count[j]);
        uniqueCacheStrategy[1]->SetCacheSize(uniqueCacheSize);
        uniqueCacheStrategy[1]->InitializeCache(uniqueCacheSize, problem.count[j]);

        //fill the two shared caches
        checkCudaErrors(cudaMemcpy2D(
                devSharedCache[i], sizeOfEachRowInCache[i],
                hostSharedCache[i], problem.count[i] * sizeof(real),
                problem.count[i] * sizeof(real), cacheSize[i], cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy2D(
                devSharedCache[j], sizeOfEachRowInCache[j],
                hostSharedCache[j], problem.count[j] * sizeof(real),
                problem.count[j] * sizeof(real), cacheSize[j], cudaMemcpyHostToDevice));

        //fill the two unique caches, or decide to compute them on-the-fly
        ACCUMULATE_TIME(preComputeTimer,
                        if (canPreComputeUniqueCache) {
                                SubHessianCalculator::preComputeUniqueCache(i, j, subProblem,
                                devUniqueCache, sizeOfEachRowInUniqueCache,
                                numOfElementEachRowInUniqueCache, param);
        )
        } else {
            if (!preComputeInHost) {
                printf("compute unique kernels on-the-fly\n");
                hessianCalculator = new DeviceHessianOnFly(subProblem, param.gamma);
            } else
                printf("use pre-compute hessian matrix in host\n");
        }
    }
}

void GpuCache::disable(int i, int j) {
    if (NULL != hessianCalculator)
        delete hessianCalculator;
    if (binary) {
        checkCudaErrors(cudaFree(devSharedCache[0]));
    } else {
        delete uniqueCacheStrategy[0];
        delete uniqueCacheStrategy[1];
        //copy the two precomputed shared caches back to host
        checkCudaErrors(cudaMemcpy2D(
                hostSharedCache[i], problem.count[i] * sizeof(real),
                devSharedCache[i], sizeOfEachRowInCache[i],
                problem.count[i] * sizeof(real), cacheSize[i], cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy2D(
                hostSharedCache[j], problem.count[j] * sizeof(real),
                devSharedCache[j], sizeOfEachRowInCache[j],
                problem.count[j] * sizeof(real), cacheSize[j], cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(devSharedCache[i]));
        checkCudaErrors(cudaFree(devSharedCache[j]));
        checkCudaErrors(cudaFree(devUniqueCache[0]));
        checkCudaErrors(cudaFree(devUniqueCache[1]));
    }
}

GpuCache::GpuCache(const SvmProblem &problem, const SVMParam &param, bool binary) :
        problem(problem), param(param), binary(binary),
        numOfElementEachRowInCache(problem.getNumOfClasses()),
        devSharedCache(problem.getNumOfClasses(), NULL),
        sizeOfEachRowInCache(problem.getNumOfClasses()),
        devUniqueCache(2),
        uniqueCacheStrategy(2),
        numOfElementEachRowInUniqueCache(2),
        sizeOfEachRowInUniqueCache(2),
        canPreComputeSharedCache(true),
        preComputeInHost(false),
        hessianCalculator(NULL) {
    if (binary) {
        printf("binary problem, use only one cache\n");
        int rowLength = problem.getNumOfSamples();
        sharedCacheStrategy.push_back(new CLATCache(rowLength));
        cacheSize.push_back(min(CACHE_SIZE * 1024 * 256 / rowLength, rowLength));
        printf("cache size = %d\n", cacheSize[0]);
        if (cacheSize[0] < problem.getNumOfSamples()) canPreComputeSharedCache = false;
        sharedCacheStrategy[0]->SetCacheSize(cacheSize[0]);
        sharedCacheStrategy[0]->InitializeCache(cacheSize[0], rowLength);
    } else {
        for (int i = 0; i < problem.getNumOfClasses(); ++i) {
            int rowLength = problem.count[i];
            sharedCacheStrategy.push_back(new CLATCache(rowLength));
            cacheSize.push_back(min(CACHE_SIZE * 1024 * 256 / rowLength / 3, rowLength));
            printf("shared cache %d size=%d, #samples in class %d=%d\n", i, cacheSize[i], i, rowLength);
            if (cacheSize[i] < problem.count[i]) canPreComputeSharedCache = false;
            sharedCacheStrategy[i]->SetCacheSize(cacheSize[i]);
            sharedCacheStrategy[i]->InitializeCache(cacheSize[i], rowLength);
            hostSharedCache.push_back(new real[cacheSize[i] * rowLength]);
        }
        if (canPreComputeSharedCache) {
            printf("cache is large enough, pre-computing shared cache\n");
            ACCUMULATE_TIME(preComputeTimer,
                            SubHessianCalculator::preComputeSharedCache(hostSharedCache, problem, param);
            )
        } else {
            if (!preComputeInHost)
                printf("compute shared kernels on-the-fly\n");
            else
                printf("use pre-compute hessian matrix in host\n");
        }
    }
//    checkCudaErrors(cudaMallocHost((void **) &hostHessianMatrix,
//                                   sizeof(float_point) * problem.getNumOfSamples() * problem.getNumOfSamples()));
//    SubHessianCalculator::preComputeAndStoreInHost(hostHessianMatrix, problem, preComputeInHost, param);
}

GpuCache::~GpuCache() {
    if (binary) {
        delete sharedCacheStrategy[0];
    } else {
        for (int i = 0; i < problem.getNumOfClasses(); ++i) {
            delete sharedCacheStrategy[i];
            delete[] hostSharedCache[i];
        }
    }
//    checkCudaErrors(cudaFreeHost(hostHessianMatrix));
}

void GpuCache::getHessianRow(int rowIndex, real *devHessianRow) {
//    printf("get row %d\n",rowIndex);
    TIMER_START(calculateKernelTimer)
    int cacheLocation;
    bool cacheFull;
    bool cacheHit;
    if (binary) {
        if (canPreComputeSharedCache) {
            cacheLocation = rowIndex;
        } else {
            cacheHit = sharedCacheStrategy[0]->GetDataFromCache(rowIndex, cacheLocation, cacheFull);
            if (!cacheHit) {
                if (cacheFull)
                    sharedCacheStrategy[0]->ReplaceExpired(rowIndex, cacheLocation, NULL);
                if (preComputeInHost) {
                    checkCudaErrors(cudaMemcpy(devSharedCache[0] + cacheLocation * numOfElementEachRowInCache[0],
                                               hostHessianMatrix + rowIndex * problem.getNumOfSamples(),
                                               sizeof(real) * problem.getNumOfSamples(),
                                               cudaMemcpyHostToDevice));
                }
                hessianCalculator->ReadRow(rowIndex,
                                           devSharedCache[0] + cacheLocation * numOfElementEachRowInCache[0],
                                           0, problem.getNumOfSamples());
            }
        }
        checkCudaErrors(cudaMemcpy(devHessianRow,
                                   devSharedCache[0] + cacheLocation * numOfElementEachRowInCache[0],
                                   sizeof(real) * problem.getNumOfSamples(),
                                   cudaMemcpyDeviceToDevice));
    } else {
        int originalLabel = subProblem->originalLabel[rowIndex]; //label in 0,1,2,3,4,...
        int originalIndex = subProblem->originalIndex[rowIndex];
        int label = 1 - (subProblem->v_nLabels[rowIndex] + 1) / 2; //map +1 -1 to 0 1
        int theOtherLabel = subProblem->label[1 - label];
        int sharedCacheStart = subProblem->start[label];
        int uniqueCacheStart = subProblem->start[1 - label];
        int sharedCacheCount = subProblem->count[label];
        int uniqueCacheCount = subProblem->count[1 - label];
        int offset = -subProblem->start[label];//TODO optimize here


        //query unique cache
        if (canPreComputeUniqueCache) {
            cacheLocation = rowIndex + offset;
        } else {
            cacheHit = uniqueCacheStrategy[label]->GetDataFromCache(rowIndex + offset, cacheLocation,
                                                                    cacheFull);
            if (!cacheHit) {
                if (cacheFull)
                    uniqueCacheStrategy[label]->ReplaceExpired(rowIndex + offset, cacheLocation, NULL);

                //unique cache position for this row
                real *tempUniqueCachePos = devUniqueCache[label] +
                                                  cacheLocation * numOfElementEachRowInUniqueCache[label];
                if (preComputeInHost)
                    checkCudaErrors(cudaMemcpy(tempUniqueCachePos,
                                               hostHessianMatrix
                                               + problem.getNumOfSamples() *
                                                 (problem.start[originalLabel] + rowIndex + offset)
                                               + problem.start[theOtherLabel],
                                               uniqueCacheCount * sizeof(real),
                                               cudaMemcpyHostToDevice));
                else
                    hessianCalculator->ReadRow(rowIndex, tempUniqueCachePos, uniqueCacheStart,
                                               uniqueCacheStart + uniqueCacheCount);
            }
        }
        checkCudaErrors(cudaMemcpy(
                devHessianRow + uniqueCacheStart,
                devUniqueCache[label] + cacheLocation * numOfElementEachRowInUniqueCache[label],
                sizeof(real) * uniqueCacheCount,
                cudaMemcpyDeviceToDevice));

        //query shared cache
        if (canPreComputeSharedCache) {
            cacheLocation = rowIndex + offset;
        } else {
            cacheHit = sharedCacheStrategy[originalLabel]->GetDataFromCache(rowIndex + offset, cacheLocation,
                                                                            cacheFull);
            if (!cacheHit) {
                if (cacheFull)
                    sharedCacheStrategy[originalLabel]->ReplaceExpired(rowIndex + offset, cacheLocation,
                                                                       NULL);
                //shared cache position
                real *tempSharedCachePos = devSharedCache[originalLabel] +
                                                  cacheLocation * numOfElementEachRowInCache[originalLabel];
                if (preComputeInHost)
                    checkCudaErrors(cudaMemcpy(tempSharedCachePos,
                                               hostHessianMatrix
                                               + problem.getNumOfSamples() *
                                                 (problem.start[originalLabel] + rowIndex + offset)
                                               + problem.start[originalLabel],
                                               sharedCacheCount * sizeof(real),
                                               cudaMemcpyHostToDevice));
                else
                    hessianCalculator->ReadRow(rowIndex, tempSharedCachePos, sharedCacheStart,
                                               sharedCacheStart + sharedCacheCount);
            }
        }
        checkCudaErrors(cudaMemcpy(
                devHessianRow + sharedCacheStart,
                devSharedCache[originalLabel] + cacheLocation * numOfElementEachRowInCache[originalLabel],
                sizeof(real) * sharedCacheCount,
                cudaMemcpyDeviceToDevice));
    }
    TIMER_STOP(calculateKernelTimer)
}
