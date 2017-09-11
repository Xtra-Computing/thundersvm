
#include <cfloat>
#include <sys/time.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include "multiSmoSolver.h"
#include "trainClassifier.h"
#include "../svm-shared/constant.h"
#include "../svm-shared/smoGPUHelper.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../svm-shared/Cache/subHessianCalculator.h"
#include "../SharedUtility/getMin.h"
#include "../SharedUtility/Timer.h"
#include "../SharedUtility/powerOfTwo.h"
#include "../SharedUtility/CudaMacro.h"
#include "multiPredictor.h"
#include <thrust/execution_policy.h>
#include <mkl.h>
#include <omp.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/system/cpp/execution_policy.h>
float hostRho, hostDiff;

void MultiSmoSolver::cpu_localSMO(const int *label, real *FValues, real *alpha, real *alphaDiff,
                                  const int *workingSet, int wsSize, float C, const float *hessianMatrixCache, int ld) {
    int sharedMem[wsSize * 3 + 2];
    int *idx4Reduce = sharedMem;
    float *fValuesI = (float *) &idx4Reduce[wsSize];
    float *fValuesJ = &fValuesI[wsSize];
    float *alphaIDiff = &fValuesJ[wsSize];
    float *alphaJDiff = &alphaIDiff[1];

//    index, f value and alpha for each instance
    float eps;
    int numOfIter = 0;
    int i, j1, j2;
    float upValue, lowValue, kJ2wsI, diff;
    int tid, wsi;
    float y, f, a;
    float *aold = new float[wsSize];
    float *kIwsI = new float[wsSize];
    float *temp_f = new float[wsSize];
    for (tid = 0; tid < wsSize; ++tid) {
        temp_f[tid] = FValues[workingSet[tid]];
        aold[tid] = alpha[workingSet[tid]];
    }
    while (1) {
//#pragma omp parallel for private(tid) schedule(guided)
        for (tid = 0; tid < wsSize; ++tid) {
            int wsi = workingSet[tid];
            int y = label[wsi];
            float f = temp_f[tid];
            float a = alpha[wsi];
            if (y > 0 && a < C || y < 0 && a > 0)
                fValuesI[tid] = f;
            else
                fValuesI[tid] = FLT_MAX;
            if (y > 0 && a > 0 || y < 0 && a < C)
                fValuesJ[tid] = -f;
            else
                fValuesJ[tid] = FLT_MAX;
        }
        i = getMin(fValuesI, idx4Reduce, wsSize);
        upValue = fValuesI[i];
        for (tid = 0; tid < wsSize; ++tid) {
            kIwsI[tid] = hessianMatrixCache[ld * i + workingSet[tid]];//K[i, wsi]
        }
        j1 = getMin(fValuesJ, idx4Reduce, wsSize);
        lowValue = -fValuesJ[j1];
        diff = lowValue - upValue;
        if (numOfIter == 0) {
            eps = max(EPS, 0.1 * diff);
        }
        if (diff < eps) {
            for (tid = 0; tid < wsSize; ++tid) {
                alphaDiff[tid] = -(alpha[workingSet[tid]] - aold[tid]) * label[workingSet[tid]];
//                printf("adiff[%d]=%f\n", tid, alphaDiff[tid]);
            }
            hostDiff = diff;
            hostRho = (lowValue + upValue) / 2;
            break;
        }

        //select j2 using second order heuristic
//#pragma omp parallel for private(tid) schedule(guided)
        for (tid = 0; tid < wsSize; ++tid) {
            int wsi = workingSet[tid];
            int y = label[wsi];
            float f = temp_f[tid];
            float a = alpha[wsi];
            if (-upValue > -f && (y > 0 && a > 0 || y < 0 && a < C)) {
                float aIJ = 1 + 1 - 2 * kIwsI[tid];
                float bIJ = -upValue + f;
                fValuesI[tid] = -bIJ * bIJ / aIJ;
            } else
                fValuesI[tid] = FLT_MAX;
//            printf("f[%d]=%f\n", tid, fValuesI[tid]);
        }
        j2 = getMin(fValuesI, idx4Reduce, wsSize);
        //update alpha
//      if (tid == i)
        y = label[workingSet[i]];
        a = alpha[workingSet[i]];
        *alphaIDiff = y > 0 ? C - a : a;
//      if (tid == j2)
        y = label[workingSet[j2]];
        a = alpha[workingSet[j2]];
        *alphaJDiff = min(y > 0 ? a : C - a, (-upValue - fValuesJ[j2]) / (1 + 1 - 2 * kIwsI[j2]));
        float l = min(*alphaIDiff, *alphaJDiff);
//        if (tid == i)
        alpha[workingSet[i]] += l * label[workingSet[i]];
//        if (tid == j2)
        alpha[workingSet[j2]] -= l * label[workingSet[j2]];
//        printf("i=%d, j1=%d, j2=%d, l=%f, ai=%f, aj2=%f\n", i, j1, j2,l,alpha[workingSet[i]], alpha[workingSet[j2]]);

        //update f
        for (tid = 0; tid < wsSize; ++tid) {
            kJ2wsI = hessianMatrixCache[ld * j2 + workingSet[tid]];//K[J2, wsi]
            temp_f[tid] -= l * (kJ2wsI - kIwsI[tid]);
        }
        numOfIter++;
    };
    delete[] aold;
    delete[] kIwsI;
    delete[] temp_f;
}


void MultiSmoSolver::solve() {
    int nrClass = problem.getNumOfClasses();

    if (model.vC.size() == 0) {//initialize C for all the binary classes
        model.vC = vector<real>(nrClass * (nrClass - 1) / 2, param.C);
    }

    //train nrClass*(nrClass-1)/2 binary models
    int k = 0;
    vector<int> prob_start(problem.start);
    for (int i = 0; i < nrClass; ++i) {
        int ci = problem.count[i];
        for (int j = i + 1; j < nrClass; ++j) {
            printf("training classifier with label %d and %d\n", i, j);
            SvmProblem subProblem = problem.getSubProblem(i, j);

            //determine the size of working set
            workingSetSize = 1024;
            if (subProblem.v_vSamples.size() < workingSetSize) {
                workingSetSize = floorPow2(subProblem.v_vSamples.size());
            }
            q = workingSetSize / 2;
            printf("q = %d, working set size = %d\n", q, workingSetSize);

            //should be called after workingSetSize is initialized
            init4Training(subProblem);

            //convert binary sub-problem to csr matrix and copy it to device
            CSRMatrix subProblemMat(subProblem.v_vSamples, subProblem.getNumOfFeatures());
            subProblemMat.copy2Dev(devVal, devRowPtr, devColInd, devSelfDot);
            nnz = subProblemMat.getNnz();

            printf("#positive ins %d, #negative ins %d\n", subProblem.count[0], subProblem.count[1]);
            int totalIter = 0;
            TIMER_START(trainingTimer)
            //start binary svm training
            for (int l = 0;; ++l) {
                workingSetIndicator = vector<int>(subProblem.getNumOfSamples(), 0);
                if (l == 0) {
                    selectWorkingSetAndPreCompute(subProblem, workingSetSize / 2, model.vC[k]);
                } else {
                    for (int m = 0; m < workingSetSize - q; ++m) {
                        workingSetIndicator[workingSet[q + m]] = 1;
                    }
                    selectWorkingSetAndPreCompute(subProblem, q / 2, model.vC[k]);
                }
                TIMER_START(iterationTimer)
                cpu_localSMO(devLabel, devYiGValue, devAlpha, devAlphaDiff, devWorkingSet, workingSetSize, model.vC[k],
                             devHessianMatrixCache, subProblem.getNumOfSamples());
                TIMER_STOP(iterationTimer)
                TIMER_START(updateGTimer)
//                updateF << < gridSize, BLOCK_SIZE >> >
//                                       (devYiGValue, devLabel, devWorkingSet, workingSetSize, devAlphaDiff, devHessianMatrixCache, subProblem.getNumOfSamples());
                int i1;
                int numOfSamples = subProblem.getNumOfSamples();
#pragma omp parallel for private(i1) schedule(guided)
                for (i1 = 0; i1 < numOfSamples; ++i1) {
                    real sumDiff = 0;
                    for (int i2 = 0; i2 < workingSetSize; ++i2) {
                        real d = devAlphaDiff[i2];
                        if (d != 0)
                            sumDiff += d * devHessianMatrixCache[i2 * numOfSamples + i1];
                    }
                    devYiGValue[i1] -= sumDiff;
                }
                TIMER_STOP(updateGTimer)
                float diff;
//                checkCudaErrors(cudaMemcpyFromSymbol(&diff, devDiff, sizeof(real), 0, cudaMemcpyDeviceToHost));
                diff = hostDiff;
                if (l % 10 == 0)
                    printf(".");
                cout.flush();
                if (diff < EPS) {
                    printf("\nup + low = %f\n", diff);
                    break;
                }
            }
            TIMER_STOP(trainingTimer)
            printf("obj = %f\n", getObjValue(subProblem.getNumOfSamples()));
            subProblemMat.freeDev(devVal, devRowPtr, devColInd, devSelfDot);
            vector<int> svIndex;
            vector<real> coef;
            real rho;



            extractModel(subProblem, svIndex, coef, rho);
            model.getModelParam(subProblem, svIndex, coef, prob_start, ci, i, j);
            model.addBinaryModel(subProblem, svIndex, coef, rho, i, j);
            k++;

            deinit4Training();
        }
    }


}

void MultiSmoSolver::init4Training(const SvmProblem &subProblem) {
    unsigned int trainingSize = subProblem.getNumOfSamples();

    workingSet = vector<int>(workingSetSize);
    devAlphaDiff = new real[workingSetSize];
    devWorkingSet = new int[workingSetSize];
    devAlpha= new real[trainingSize];
    devYiGValue= new real[trainingSize];
//    devLabel= new int[trainingSize];
//    devWorkingSetIndicator= new int[trainingSize];
//    checkCudaErrors(cudaMallocManaged((void **) &devAlphaDiff, sizeof(real) * workingSetSize));
//    checkCudaErrors(cudaMallocManaged((void **) &devWorkingSet, sizeof(int) * workingSetSize));
//
//    checkCudaErrors(cudaMallocManaged((void **) &devAlpha, sizeof(real) * trainingSize));
//    checkCudaErrors(cudaMallocManaged((void **) &devYiGValue, sizeof(real) * trainingSize));
//    checkCudaErrors(cudaMallocManaged((void **) &devLabel, sizeof(int) * trainingSize));
//    checkCudaErrors(cudaMallocManaged((void **) &devWorkingSetIndicator, sizeof(int) * trainingSize));

    memset(devAlpha, 0, sizeof(real) * trainingSize);
//    checkCudaErrors(cudaMemset(devAlpha, 0, sizeof(real) * trainingSize));
    vector<real> negatedLabel(trainingSize);
    for (int i = 0; i < trainingSize; ++i) {
//        negatedLabel[i] = -subProblem.v_nLabels[i];
        devYiGValue[i] = -subProblem.v_nLabels[i];
    }
//    checkCudaErrors(cudaMemcpy(devYiGValue, negatedLabel.data(), sizeof(real) * trainingSize,
//                               cudaMemcpyHostToDevice));
//    checkCudaErrors(
//            cudaMemcpy(devLabel, subProblem.v_nLabels.data(), sizeof(int) * trainingSize, cudaMemcpyHostToDevice));
    devLabel = subProblem.v_nLabels.data();

    InitSolver(trainingSize);//initialise base solver

    devHessianMatrixCache = new real[workingSetSize * trainingSize];
//    checkCudaErrors(cudaMallocManaged((void **) &devHessianMatrixCache, sizeof(real) * workingSetSize * trainingSize));

    for (int j = 0; j < trainingSize; ++j) {
//        hessianDiag[j] = 1;//assume using RBF kernel
        devHessianDiag[j] = 1;//assume using RBF kernel
    }
//    checkCudaErrors(
//            cudaMemcpy(devHessianDiag, hessianDiag, sizeof(real) * trainingSize, cudaMemcpyHostToDevice));
    devFValue4Sort = new real[trainingSize];
    devIdx4Sort = new int[trainingSize];
//    checkCudaErrors(cudaMallocManaged((void **) &devFValue4Sort, sizeof(real) * trainingSize));
//    checkCudaErrors(cudaMallocManaged((void **) &devIdx4Sort, sizeof(int) * trainingSize));

}

void MultiSmoSolver::deinit4Training() {
    delete[] devAlphaDiff;
    delete[] devWorkingSet;
    delete[]devAlpha;
    delete[]devYiGValue;
//    delete[]devLabel;
//    delete[]devWorkingSetIndicator;
//    checkCudaErrors(cudaFree(devAlphaDiff));
//    checkCudaErrors(cudaFree(devWorkingSet));
//
//    checkCudaErrors(cudaFree(devAlpha));
//    checkCudaErrors(cudaFree(devYiGValue));
//    checkCudaErrors(cudaFree(devLabel));
//    checkCudaErrors(cudaFree(devWorkingSetIndicator));

    DeInitSolver();

    delete[] devHessianMatrixCache;
    delete[] devFValue4Sort;
    delete[] devIdx4Sort;
//    checkCudaErrors(cudaFree(devHessianMatrixCache));
//    checkCudaErrors(cudaFree(devFValue4Sort));
//    checkCudaErrors(cudaFree(devIdx4Sort));
}

void MultiSmoSolver::extractModel(const SvmProblem &subProblem, vector<int> &svIndex, vector<real> &coef,
                                  real &rho) const {
    const unsigned int trainingSize = subProblem.getNumOfSamples();
    vector<real> alpha(trainingSize);
    const vector<int> &label = subProblem.v_nLabels;
    checkCudaErrors(cudaMemcpy(alpha.data(), devAlpha, sizeof(real) * trainingSize, cudaMemcpyHostToHost));
    for (int i = 0; i < trainingSize; ++i) {
        if (alpha[i] != 0) {
            coef.push_back(label[i] * alpha[i]);
            svIndex.push_back(i);

        }
    }
//    checkCudaErrors(cudaMemcpyFromSymbol(&rho, devRho, sizeof(real), 0, cudaMemcpyDeviceToHost));
    rho = hostRho;
    printf("# of SV %lu\nbias = %f\n", svIndex.size(), rho);
}

MultiSmoSolver::MultiSmoSolver(const SvmProblem &problem, SvmModel &model, const SVMParam &param) :
        problem(problem), model(model), param(param) {
}

MultiSmoSolver::~MultiSmoSolver() {
}

void
MultiSmoSolver::selectWorkingSetAndPreCompute(const SvmProblem &subProblem, uint numOfSelectPairs, real penaltyC) {
    uint numOfSamples = subProblem.getNumOfSamples();
    uint oldSize = workingSetSize / 2 - numOfSelectPairs;
    TIMER_START(selectTimer)
    vector<int> oldWorkingSet = workingSet;

    devWorkingSetIndicator = workingSetIndicator.data();
//    checkCudaErrors(cudaMemcpy(devWorkingSetIndicator, workingSetIndicator.data(), sizeof(int) * numOfSamples,
//                               cudaMemcpyHostToDevice));

//#pragma omp parallel for private(i) schedule(guided)
    for (int i = 0; i < numOfSamples; ++i) {
        if (i < numOfSamples) {
            real y = devLabel[i];
            real a = devAlpha[i];
            if (devWorkingSetIndicator[i] == 1)
                devFValue4Sort[i] = -FLT_MAX;
            else if (y > 0 && a < penaltyC || y < 0 && a > 0)
                devFValue4Sort[i] = -devYiGValue[i];
            else
                devFValue4Sort[i] = -FLT_MAX + 1;
            devIdx4Sort[i] = i;
        }
    }
    thrust::sort_by_key(thrust::cpp::par, devFValue4Sort, devFValue4Sort + numOfSamples, devIdx4Sort,
                        thrust::greater<float>());
    checkCudaErrors(cudaMemcpy(workingSet.data() + oldSize * 2, devIdx4Sort, sizeof(int) * numOfSelectPairs,
                               cudaMemcpyHostToHost));
    for (int i = 0; i < numOfSelectPairs; ++i) {
        workingSetIndicator[workingSet[oldSize * 2 + i]] = 1;
    }
    checkCudaErrors(cudaMemcpy(devWorkingSetIndicator, workingSetIndicator.data(), sizeof(int) * numOfSamples,
                               cudaMemcpyHostToHost));
//#pragma omp parallel for private(i) schedule(guided)
    for (int i = 0; i < numOfSamples; ++i) {
        if (i < numOfSamples) {
            real y = devLabel[i];
            real a = devAlpha[i];
            if (devWorkingSetIndicator[i] == 1)
                devFValue4Sort[i] = -FLT_MAX;
            else if (y > 0 && a > 0 || y < 0 && a < penaltyC)
                devFValue4Sort[i] = devYiGValue[i];
            else
                devFValue4Sort[i] = -FLT_MAX + 1;
            devIdx4Sort[i] = i;
        }
    }
    thrust::sort_by_key(thrust::cpp::par, devFValue4Sort, devFValue4Sort + numOfSamples, devIdx4Sort,
                        thrust::greater<float>());
    checkCudaErrors(
            cudaMemcpy(workingSet.data() + oldSize * 2 + numOfSelectPairs, devIdx4Sort, sizeof(int) * numOfSelectPairs,
                       cudaMemcpyHostToHost));

    //get pairs from last working set
    for (int i = 0; i < oldSize * 2; ++i) {
        workingSet[i] = oldWorkingSet[numOfSelectPairs * 2 + i];
    }
    checkCudaErrors(cudaMemcpy(devWorkingSet, workingSet.data(), sizeof(int) * workingSetSize,cudaMemcpyHostToHost));
    TIMER_STOP(selectTimer)

    //move old kernel values to get space
    checkCudaErrors(cudaMemcpy(devHessianMatrixCache,
                               devHessianMatrixCache + numOfSamples * numOfSelectPairs * 2,
                               sizeof(real) * numOfSamples * oldSize * 2,
                               cudaMemcpyHostToHost));
    vector<vector<KeyValue> > computeSamples;
    for (int i = 0; i < numOfSelectPairs * 2; ++i) {
        computeSamples.push_back(subProblem.v_vSamples[workingSet[oldSize * 2 + i]]);
    }
    TIMER_START(preComputeTimer)
    //preCompute kernel values of new selected instances
    CSRMatrix workingSetMat(computeSamples, subProblem.getNumOfFeatures());
    real *devWSVal;
    int *devWSColInd;
    int *devWSRowPtr;
    real *devWSSelfDot;
    workingSetMat.copy2Dev(devWSVal, devWSRowPtr, devWSColInd, devWSSelfDot);
    MKL_INT m, n, lda;
    m = numOfSelectPairs * 2;
    n = subProblem.getNumOfFeatures();
    lda = n;
    int info[0];
    float *a = new float[m * n];
    float *trans_a = new float[m * n];
    MKL_INT job[6];
    job[0] = 1;
    job[1] = 0;
    job[2] = 0;
    job[3] = 2;
    job[4] = m * n;
    job[5] = 1;
    mkl_sdnscsr(job, &m, &n, a, &lda, devWSVal, devWSColInd, devWSRowPtr, info);
    assert(*info == 0);
    mkl_somatcopy('r', 't', m, n, 1, a, n, trans_a, m);
    MKL_INT k;
    float alpha(1), beta(0);
    m = numOfSamples;
    n = numOfSelectPairs * 2;
    k = subProblem.getNumOfFeatures();
    char transa = 'n';
    char matdesca[6];
    matdesca[0] = 'g';
    matdesca[1] = 'l';
    matdesca[2] = 'n';
    matdesca[3] = 'c';
    real *kernel = new real[m * n];
    mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdesca, devVal, devColInd, devRowPtr, devRowPtr + 1, trans_a, &n, &beta,
               kernel, &n);
    mkl_somatcopy('r', 't', m, n, 1, kernel, n, devHessianMatrixCache + numOfSamples * oldSize * 2, m);
#pragma omp parallel for private(i) schedule(guided)
    for (int i = oldSize * 2; i < workingSetSize; ++i) {
        for (int j = 0; j < m; ++j) {
            devHessianMatrixCache[i * m + j] = expf(-param.gamma * (-2 * devHessianMatrixCache[i * m + j] +
                                                                    devWSSelfDot[i - oldSize * 2] + devSelfDot[j]));
        }
    }

    delete[] a;
    delete[] trans_a;
    delete[] kernel;
    TIMER_STOP(preComputeTimer)
}

real MultiSmoSolver::getObjValue(int numOfSamples) const {
    //the function should be called before deinit4Training
    vector<real> f(numOfSamples);
    vector<real> alpha(numOfSamples);
    vector<int> y(numOfSamples);
    cudaMemcpy(f.data(), devYiGValue, sizeof(real) * numOfSamples,cudaMemcpyHostToHost);
    cudaMemcpy(alpha.data(), devAlpha, sizeof(real) * numOfSamples,cudaMemcpyHostToHost);
    cudaMemcpy(y.data(), devLabel, sizeof(int) * numOfSamples,cudaMemcpyHostToHost);
    real obj = 0;
    for (int i = 0; i < numOfSamples; ++i) {
        obj -= alpha[i];
    }
    for (int i = 0; i < numOfSamples; ++i) {
        obj += 0.5 * alpha[i] * y[i] * (f[i] + y[i]);
    }
    return obj;
}

int MultiSmoSolver::getMin(float *values, int *index, int size) {
    float min = FLT_MAX;
    int min_index = 0;
    for (int i = 0; i < size; ++i) {
        if (values[i] < min) {
            min = values[i];
            min_index = i;
        }
    }
    return min_index;
}

