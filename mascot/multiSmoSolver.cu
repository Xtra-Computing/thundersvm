//
// Created by ss on 16-12-14.
//

#include <zconf.h>
#include <sys/time.h>
#include "multiSmoSolver.h"
#include "../svm-shared/constant.h"
#include "cuda_runtime.h"
#include "trainingFunction.h"
#include "../svm-shared/smoGPUHelper.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../SharedUtility/Timer.h"

void MultiSmoSolver::solve() {
    int nrClass = problem.getNumOfClasses();
    //train nrClass*(nrClass-1)/2 binary models
    int k = 0;
    for (int i = 0; i < nrClass; ++i) {
        for (int j = i + 1; j < nrClass; ++j) {
            printf("training classifier with label %d and %d\n", i, j);
            SvmProblem subProblem = problem.getSubProblem(i, j);
            init4Training(subProblem);
            cache.enable(i, j, subProblem);
            int maxIter = (subProblem.getNumOfSamples() > INT_MAX / ITERATION_FACTOR
                           ? INT_MAX
                           : ITERATION_FACTOR * subProblem.getNumOfSamples()) * 4;
            int numOfIter;
            TIMER_START(iterationTimer)
            for (numOfIter = 0; numOfIter < maxIter && !iterate(subProblem, param.C); numOfIter++) {
                if (numOfIter % 1000 == 0 && numOfIter != 0) {
                    std::cout << ".";
                    std::cout.flush();
                }
            }
            TIMER_STOP(iterationTimer)
            cache.disable(i, j);

            cout << "# of iteration: " << numOfIter << endl;
            vector<int> svIndex;
            vector<float_point> coef;
            float_point rho;
            extractModel(subProblem, svIndex, coef, rho);

            //measure training errors and prediction errors

            model.addBinaryModel(subProblem, svIndex, coef, rho, i, j);
            k++;
            deinit4Training();
        }
    }
}

bool MultiSmoSolver::iterate(SvmProblem &subProblem, float_point C) {
    int trainingSize = subProblem.getNumOfSamples();

    TIMER_START(selectTimer)
    SelectFirst(trainingSize, C);
    SelectSecond(trainingSize, C);
    TIMER_STOP(selectTimer)


    IdofInstanceTwo = int(hostBuffer[0]);

    //get kernel value K(Sample1, Sample2)
    float_point fKernelValue = 0;
    float_point fMinLowValue;
    fMinLowValue = hostBuffer[1];
    fKernelValue = hostBuffer[2];


    cache.getHessianRow(IdofInstanceTwo,devHessianInstanceRow2);


    lowValue = -hostBuffer[3];
    //check if the problem is converged
    if (upValue + lowValue <= EPS) {
        //cout << upValue << " : " << lowValue << endl;
        //m_pGPUCache->PrintCachingStatistics();
        return true;
    }

    float_point fY1AlphaDiff, fY2AlphaDiff;
    float_point fMinValue = -upValue;
    TIMER_START(updateAlphaTimer)
    UpdateTwoWeight(fMinLowValue, fMinValue, IdofInstanceOne, IdofInstanceTwo, fKernelValue,
                    fY1AlphaDiff, fY2AlphaDiff, subProblem.v_nLabels.data(), C);

    TIMER_STOP(updateAlphaTimer)
    TIMER_START(updateGTimer)
    UpdateYiGValue(trainingSize, fY1AlphaDiff, fY2AlphaDiff);
    TIMER_STOP(updateGTimer)

    return false;
}

void MultiSmoSolver::init4Training(const SvmProblem &subProblem) {
    unsigned int trainingSize = subProblem.getNumOfSamples();

    checkCudaErrors(cudaMalloc((void **) &devAlpha, sizeof(float_point) * trainingSize));
    checkCudaErrors(cudaMalloc((void **) &devYiGValue, sizeof(float_point) * trainingSize));
    checkCudaErrors(cudaMalloc((void **) &devLabel, sizeof(int) * trainingSize));

    checkCudaErrors(cudaMemset(devAlpha, 0, sizeof(float_point) * trainingSize));
    vector<float_point> negatedLabel(trainingSize);
    for (int i = 0; i < trainingSize; ++i) {
    	negatedLabel[i] = -subProblem.v_nLabels[i];
    }
    checkCudaErrors(cudaMemcpy(devYiGValue, negatedLabel.data(), sizeof(float_point) * trainingSize,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devLabel, subProblem.v_nLabels.data(), sizeof(int) * trainingSize, cudaMemcpyHostToDevice));

    InitSolver(trainingSize);//initialise base solver

    checkCudaErrors(cudaMalloc((void **) &devHessianInstanceRow1, sizeof(float_point) * trainingSize));
    checkCudaErrors(cudaMalloc((void **) &devHessianInstanceRow2, sizeof(float_point) * trainingSize));

//    int cacheSize = CACHE_SIZE * 1024 * 1024 / 4 / trainingSize;
//    gpuCache = new CLATCache(trainingSize);
//    gpuCache->SetCacheSize(cacheSize);
//    gpuCache->InitializeCache(cacheSize, trainingSize);
//    size_t sizeOfEachRowInCache;
//    checkCudaErrors(
//            cudaMallocPitch((void **) &devHessianMatrixCache, &sizeOfEachRowInCache, trainingSize * sizeof(float_point),
//                            cacheSize));
//    //temp memory for reading result to cache
//    numOfElementEachRowInCache = sizeOfEachRowInCache / sizeof(float_point);
//    if (numOfElementEachRowInCache != trainingSize) {
//        cout << "cache memory aligned to: " << numOfElementEachRowInCache
//             << "; number of the training instances is: " << trainingSize << endl;
//    }
//    cout << "cache size v.s. ins is " << cacheSize << " v.s. " << trainingSize << endl;
//
//    checkCudaErrors(cudaMemset(devHessianMatrixCache, 0, cacheSize * sizeOfEachRowInCache));

//    hessianCalculator = new DeviceHessianOnFly(subProblem, param.gamma);
//    hessianCalculator->GetHessianDiag("", trainingSize, hessianDiag);
    for (int j = 0; j < trainingSize; ++j) {
        hessianDiag[j] = 1;//assume using RBF kernel
    }
    checkCudaErrors(cudaMemcpy(devHessianDiag, hessianDiag, sizeof(float_point) * trainingSize, cudaMemcpyHostToDevice));
}

void MultiSmoSolver::deinit4Training() {
    checkCudaErrors(cudaFree(devAlpha));
    checkCudaErrors(cudaFree(devYiGValue));
    checkCudaErrors(cudaFree(devLabel));

    DeInitSolver();

//    checkCudaErrors(cudaFree(devHessianMatrixCache));
    checkCudaErrors(cudaFree(devHessianInstanceRow1));
    checkCudaErrors(cudaFree(devHessianInstanceRow2));
}

int MultiSmoSolver::getHessianRow(int rowIndex) {
    int cacheLocation;
    bool cacheFull = false;
    bool cacheHit = gpuCache->GetDataFromCache(rowIndex,cacheLocation,cacheFull);
    if (!cacheHit) {
        if (cacheFull)
            gpuCache->ReplaceExpired(rowIndex, cacheLocation, NULL);
        hessianCalculator->ReadRow(rowIndex, devHessianMatrixCache + cacheLocation * numOfElementEachRowInCache,0 ,100);
    }
    return cacheLocation * numOfElementEachRowInCache;
}


void MultiSmoSolver::extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<float_point> &coef,
                                  float_point &rho) const {
    const unsigned int trainingSize = subProblem.getNumOfSamples();
    vector<float_point> alpha(trainingSize);
    const vector<int> &label = subProblem.v_nLabels;
    checkCudaErrors(cudaMemcpy(alpha.data(), devAlpha, sizeof(float_point) * trainingSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < trainingSize; ++i) {
        if (alpha[i] != 0) {
            coef.push_back(label[i] * alpha[i]);
            svIndex.push_back(i);
        }
    }
    rho = (lowValue - upValue) / 2;
    printf("# of SV %lu\nbias = %f\n", svIndex.size(), rho);
}
