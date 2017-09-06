
#include "smoGPUHelper.h"
#include "gpu_global_utility.h"
#include "../SharedUtility/getMin.h"
#include "../SharedUtility/CudaMacro.h"

#include <float.h>
#include <cstdio>

/* *
 /*
 * @brief: kernel funciton for getting minimum value within a block
 * @param: pfYiFValue: a set of value = y_i * gradient of subjective function
 * @param: pfAlpha:	   a set of alpha related to training samples
 * @param: pnLabel:	   a set of labels related to training samples
 * @param: nNumofTrainingSamples: the number of training samples
 * @param: pfBlockMin: the min value of this block (function result)
 * @param: pnBlockMinGlobalKey: the index of the min value of this block
 */
__global__ void GetBlockMinYiGValue(real *pfYiFValue, real *pfAlpha, int *pnLabel, real fPCost,
                                    int nNumofTrainingSamples, real *pfBlockMin, int *pnBlockMinGlobalKey) {
    __shared__ real fTempLocalYiFValue[BLOCK_SIZE];
    __shared__ int nTempLocalKeys[BLOCK_SIZE];

    int nGlobalIndex;
    int nThreadId = threadIdx.x;
    //global index for thread
    nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    fTempLocalYiFValue[nThreadId] = FLT_MAX;
    if (nGlobalIndex < nNumofTrainingSamples) {
        real fAlpha;
        int nLabel;
        fAlpha = pfAlpha[nGlobalIndex];
        nLabel = pnLabel[nGlobalIndex];
        //fill yi*GValue in a block
        if ((nLabel > 0 && fAlpha < fPCost) || (nLabel < 0 && fAlpha > 0)) {
            //I_0 is (fAlpha > 0 && fAlpha < fCostP). This condition is covered by the following condition
            //index set I_up
            fTempLocalYiFValue[nThreadId] = pfYiFValue[nGlobalIndex];
            nTempLocalKeys[nThreadId] = nGlobalIndex;
        }
    }
    __syncthreads();    //synchronize threads within a block, and start to do reduce

    GetMinValueOriginal(fTempLocalYiFValue, nTempLocalKeys);

    if (nThreadId == 0) {
        int nBlockId = blockIdx.y * gridDim.x + blockIdx.x;
        pfBlockMin[nBlockId] = fTempLocalYiFValue[0];
        pnBlockMinGlobalKey[nBlockId] = nTempLocalKeys[0];
    }
}

/*
 * @brief: for selecting the second sample to optimize
 * @param: pfYiFValue: the gradient of data samples
 * @param: pfAlpha: alpha values for samples
 * @param: fNCost: the cost of negative sample (i.e., the C in SVM)
 * @param: pfDiagHessian: the diagonal of Hessian Matrix
 * @param: pfHessianRow: a Hessian row of sample one
 * @param: fMinusYiUpValue: -yi*gradient of sample one
 * @param: fUpValueKernel: self dot product of sample one
 * @param: pfBlockMin: minimum value of each block (the output of this kernel)
 * @param: pnBlockMinGlobalKey: the key of each block minimum value (the output of this kernel)
 * @param: pfBlockMinYiFValue: the block minimum gradient (the output of this kernel. for convergence check)
 */
__global__ void GetBlockMinLowValue(real *pfYiFValue, real *pfAlpha, int *pnLabel, real fNCost,
                                    int nNumofTrainingSamples, real *pfDiagHessian, real *pfHessianRow,
                                    real fMinusYiUpValue, real fUpValueKernel, real *pfBlockMin,
                                    int *pnBlockMinGlobalKey, real *pfBlockMinYiFValue) {
    __shared__ int nTempKey[BLOCK_SIZE];
    __shared__ real fTempMinValues[BLOCK_SIZE];

    int nThreadId = threadIdx.x;
    int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread

    fTempMinValues[nThreadId] = FLT_MAX;
    //fTempMinYiFValue[nThreadId] = FLT_MAX;

    //fill data (-b_ij * b_ij/a_ij) into a block
    real fYiGValue;
    real fBeta;
    int nReduce = NOREDUCE;
    real fAUp_j;
    real fBUp_j;

    if (nGlobalIndex < nNumofTrainingSamples) {
        real fUpValue = fMinusYiUpValue;
        fYiGValue = pfYiFValue[nGlobalIndex];
        real fAlpha = pfAlpha[nGlobalIndex];

        nTempKey[nThreadId] = nGlobalIndex;

        int nLabel = pnLabel[nGlobalIndex];
        /*************** calculate b_ij ****************/
        //b_ij = -Gi + Gj in paper, but b_ij = -Gi + y_j * Gj in the code of libsvm. Here we follow the code of libsvm
        fBUp_j = fUpValue + fYiGValue;

        if (((nLabel > 0) && (fAlpha > 0)) ||
            ((nLabel < 0) && (fAlpha < fNCost))
                ) {
            fAUp_j = fUpValueKernel + pfDiagHessian[nGlobalIndex] - 2 * pfHessianRow[nGlobalIndex];

            if (fAUp_j <= 0) {
                fAUp_j = TAU;
            }

            if (fBUp_j > 0) {
                nReduce = REDUCE1 | REDUCE0;
            } else
                nReduce = REDUCE0;

            //for getting optimized pair
            //fBeta = -(fBUp_j * fBUp_j / fAUp_j);
            fBeta = __fdividef(__powf(fBUp_j, 2.f), fAUp_j);
            fBeta = -fBeta;
            //fTempMinYiFValue[nThreadId] = -fYiGValue;
        }
    }

    if ((nReduce & REDUCE0) != 0) {
        fTempMinValues[threadIdx.x] = -fYiGValue;
    }
    __syncthreads();
    GetMinValueOriginal(fTempMinValues);
    int nBlockId;
    if (nThreadId == 0) {
        nBlockId = blockIdx.y * gridDim.x + blockIdx.x;
        pfBlockMinYiFValue[nBlockId] = fTempMinValues[0];
    }

    fTempMinValues[threadIdx.x] = (((nReduce & REDUCE1) != 0) ? fBeta : FLT_MAX);

    //block level reduce
    __syncthreads();
    GetMinValueOriginal(fTempMinValues, nTempKey);
    __syncthreads();

    if (nThreadId == 0) {
        pfBlockMin[nBlockId] = fTempMinValues[0];
        pnBlockMinGlobalKey[nBlockId] = nTempKey[0];
    }
}

/*
 * @brief: kernel function for getting the minimum value in a set of block min values
 * @param: pfBlockMin: a set of min value returned from block level reducer
 * @param: pnBlockMinKey: a set of indices for block min (i.e., each block min value has a global index)
 * @param: nNumofBlock:	  the number of blocks
 * @param: pfMinValue:	  a pointer to global min value (the result of this function)
 * @param: pnMinKey:	  a pointer to the index of the global min value (the result of this function)
 */
__global__ void GetGlobalMin(real *pfBlockMin, int *pnBlockMinKey, int nNumofBlock,
                             real *pfYiFValue, real *pfHessianRow, real *pfTempKeyValue) {
    __shared__ int nTempKey[BLOCK_SIZE];
    __shared__ real pfTempMin[BLOCK_SIZE];
    int nThreadId = threadIdx.x;

    if (nThreadId < nNumofBlock) {
        nTempKey[nThreadId] = pnBlockMinKey[nThreadId];
        pfTempMin[nThreadId] = pfBlockMin[nThreadId];
    } else {
        //nTempKey[nThreadId] = pnBlockMinKey[nThreadId];
        pfTempMin[nThreadId] = FLT_MAX;
    }

    //if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
    if (nNumofBlock > BLOCK_SIZE) {
        real fTempMin = pfTempMin[nThreadId];
        int nTempMinKey = nTempKey[nThreadId];
        for (int i = nThreadId + BLOCK_SIZE; i < nNumofBlock; i += blockDim.x) {
            real fTempBlockMin = pfBlockMin[i];
            if (fTempBlockMin < fTempMin) {
                //store the minimum value and the corresponding key
                fTempMin = fTempBlockMin;
                nTempMinKey = pnBlockMinKey[i];
            }
        }
        nTempKey[nThreadId] = nTempMinKey;
        pfTempMin[nThreadId] = fTempMin;
    }
    __syncthreads();    //wait until the thread within the block

    GetMinValueOriginal(pfTempMin, nTempKey);

    if (nThreadId == 0) {
        *(pfTempKeyValue) = (real) nTempKey[0];
        if (pfYiFValue != NULL) {
            *(pfTempKeyValue + 1) = pfYiFValue[nTempKey[0]];//pfTempMin[0];
        } else {
            *(pfTempKeyValue + 1) = pfTempMin[0];
        }

        if (pfHessianRow != NULL) {
            *(pfTempKeyValue + 2) = pfHessianRow[nTempKey[0]];
        }
    }
}

/*
 * @brief: kernel function for getting the minimum value in a set of block min values
 * @param: pfBlockMin: a set of min value returned from block level reducer
 * @param: pnBlockMinKey: a set of indices for block min (i.e., each block min value has a global index)
 * @param: nNumofBlock:	  the number of blocks
 * @param: pfMinValue:	  a pointer to global min value (the result of this function)
 * @param: pnMinKey:	  a pointer to the index of the global min value (the result of this function)
 */
__global__ void GetGlobalMin(real *pfBlockMin, int nNumofBlock, real *pfTempKeyValue) {
    __shared__ real pfTempMin[BLOCK_SIZE];
    int nThreadId = threadIdx.x;

    pfTempMin[nThreadId] = ((nThreadId < nNumofBlock) ? pfBlockMin[nThreadId] : FLT_MAX);

    //if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
    if (nNumofBlock > BLOCK_SIZE) {
        real fTempMin = pfTempMin[nThreadId];
        for (int i = nThreadId + BLOCK_SIZE; i < nNumofBlock; i += blockDim.x) {
            real fTempBlockMin = pfBlockMin[i];
            fTempMin = (fTempBlockMin < fTempMin) ? fTempBlockMin : fTempMin;
        }
        pfTempMin[nThreadId] = fTempMin;
    }
    __syncthreads();    //wait until the thread within the block

    GetMinValueOriginal(pfTempMin);
    __syncthreads();

    if (nThreadId == 0) {
        *(pfTempKeyValue + 3) = pfTempMin[0];
    }
}

/*
 * @brief: update gradient values for all samples
 * @param: pfYiFValue: the gradient of samples (input and output of this kernel)
 * @param: pfHessianRow1: the Hessian row of sample one
 * @param: pfHessianRow2: the Hessian row of sample two
 * @param: fY1AlphaDiff: the difference of old and new alpha of sample one
 * @param: fY2AlphaDiff: the difference of old and new alpha of sample two
 */
__global__ void
UpdateYiFValueKernel(real *pfAlpha, real *pDevBuffer, real *pfYiFValue, real *pfHessianRow1,
                     real *pfHessianRow2,
                     real fY1AlphaDiff, real fY2AlphaDiff, int nNumofTrainingSamples) {
    if (threadIdx.x < 2) {
        int nTemp = int(pDevBuffer[threadIdx.x * 2]);
        pfAlpha[nTemp] = pDevBuffer[threadIdx.x * 2 + 1];
        //nTemp = int(pDevBuffer[2]);
        //pfAlpha[nTemp] = pDevBuffer[3];
    }
    __syncthreads();
    real fsY1AlphaDiff;
    fsY1AlphaDiff = fY1AlphaDiff;
    real fsY2AlphaDiff;
    fsY2AlphaDiff = fY2AlphaDiff;

    int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread

    if (nGlobalIndex < nNumofTrainingSamples) {
        //update YiFValue
        pfYiFValue[nGlobalIndex] += (fsY1AlphaDiff * pfHessianRow1[nGlobalIndex] +
                                     fsY2AlphaDiff * pfHessianRow2[nGlobalIndex]);
    }

}

/**
 * local SMO in one block
 * @param label: label for all instances in training set
 * @param FValues: f value for all instances in training set
 * @param alpha: alpha for all instances in training set
 * @param alphaDiff: difference of each alpha in working set after local SMO
 * @param workingSet: index of each instance in training set
 * @param wsSize: size of working set
 * @param C: C parameter in SVM
 * @param hessianMatrixCache: |working set| * |training set| kernel matrix, row major
 * @param ld: number of instances each row in hessianMatrixCache
 */
__global__ void localSMO(const int *label, real *FValues, real *alpha, real *alphaDiff,
                         const int *workingSet, int wsSize, float C, const float *hessianMatrixCache, int ld) {

    //allocate shared memory
    extern __shared__ int sharedMem[];
    int *idx4Reduce = sharedMem;
    float *fValuesI = (float *) &idx4Reduce[wsSize];
    float *fValuesJ = &fValuesI[wsSize];
    float *alphaIDiff = &fValuesJ[wsSize];
    float *alphaJDiff = &alphaIDiff[1];

    //index, f value and alpha for each instance
    int tid = threadIdx.x;
    int wsi = workingSet[tid];
    float y = label[wsi];
    float f = FValues[wsi];
    float a = alpha[wsi];
    float aold = a;
    __syncthreads();
    float eps;
    int numOfIter = 0;
    while (1) {
        //select fUp and fLow
        if (y > 0 && a < C || y < 0 && a > 0)
            fValuesI[tid] = f;
        else
            fValuesI[tid] = FLT_MAX;
        if (y > 0 && a > 0 || y < 0 && a < C)
            fValuesJ[tid] = -f;
        else
            fValuesJ[tid] = FLT_MAX;
        int i = getBlockMin(fValuesI, idx4Reduce);
        float upValue = fValuesI[i];
        float kIwsI = hessianMatrixCache[ld * i + wsi];//K[i, wsi]
        __syncthreads();
        int j1 = getBlockMin(fValuesJ, idx4Reduce);
        float lowValue = -fValuesJ[j1];

        float diff = lowValue - upValue;
        if (numOfIter == 0) {
            if (tid == 0)
                devDiff = diff;
            eps = max(EPS, 0.1 * diff);
        }

        if (diff < eps) {
            alpha[wsi] = a;
            alphaDiff[tid] = -(a - aold) * y;
            if (tid == 0) {
                devRho = (lowValue + upValue) / 2;
            }
            break;
        }
        __syncthreads();

        //select j2 using second order heuristic
        if (-upValue > -f && (y > 0 && a > 0 || y < 0 && a < C)) {
            float aIJ = 1 + 1 - 2 * kIwsI;
            float bIJ = -upValue + f;
            fValuesI[tid] = -bIJ * bIJ / aIJ;
        } else
            fValuesI[tid] = FLT_MAX;
        int j2 = getBlockMin(fValuesI, idx4Reduce);

        //update alpha
        if (tid == i)
            *alphaIDiff = y > 0 ? C - a : a;
        if (tid == j2)
            *alphaJDiff = min(y > 0 ? a : C - a, (-upValue - fValuesJ[j2]) / (1 + 1 - 2 * kIwsI));
        __syncthreads();
        float l = min(*alphaIDiff, *alphaJDiff);

        if (tid == i)
            a += l * y;
        if (tid == j2)
            a -= l * y;

        //update f
        float kJ2wsI = hessianMatrixCache[ld * j2 + wsi];//K[J2, wsi]
        f -= l * (kJ2wsI - kIwsI);
        numOfIter++;
    }
};

/**
 * update f values using alpha diff
 * @param FValues: f values for all instances in training set
 * @param label: label for all instances in training set
 * @param workingSet: index of each instance in working set
 * @param wsSize: size of working set
 * @param alphaDiff: difference of alpha in working set
 * @param hessianMatrixCache: |working set| * |training set| kernel matrix, row major
 * @param numOfSamples
 */
__global__ void
updateF(real *FValues, const int *label, const int *workingSet, int wsSize, const real *alphaDiff,
        const real *hessianMatrixCache, int numOfSamples) {
    int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
    if (nGlobalIndex < numOfSamples) {
        real sumDiff = 0;
        for (int i = 0; i < wsSize; ++i) {
            real d = alphaDiff[i];
            if (d != 0)
                sumDiff += d * hessianMatrixCache[i * numOfSamples + nGlobalIndex];
        }
        FValues[nGlobalIndex] -= sumDiff;
    }
}

__global__ void getFUpValues(const real *FValues, const real *alpha, const int *labels,
                             int numOfSamples, real C, real *FValue4Sort, int *Idx4Sort, const int *wsIndicator) {
    int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
    if (nGlobalIndex < numOfSamples) {
        real y = labels[nGlobalIndex];
        real a = alpha[nGlobalIndex];
//        printf("[%d]%d",nGlobalIndex, wsIndicator[nGlobalIndex]);
        if (wsIndicator[nGlobalIndex] == 1)
            FValue4Sort[nGlobalIndex] = -FLT_MAX;
        else if (y > 0 && a < C || y < 0 && a > 0)
            FValue4Sort[nGlobalIndex] = -FValues[nGlobalIndex];
        else
            FValue4Sort[nGlobalIndex] = -FLT_MAX + 1;
        Idx4Sort[nGlobalIndex] = nGlobalIndex;
    }
}

__global__ void getFLowValues(const real *FValues, const real *alpha, const int *labels,
                              int numOfSamples, real C, real *FValue4Sort, int *Idx4Sort, const int *wsIndicator) {
    int nGlobalIndex = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;//global index for thread
    if (nGlobalIndex < numOfSamples) {
        real y = labels[nGlobalIndex];
        real a = alpha[nGlobalIndex];
//        printf("[%d]%d",nGlobalIndex, wsIndicator[nGlobalIndex]);
        if (wsIndicator[nGlobalIndex] == 1)
            FValue4Sort[nGlobalIndex] = -FLT_MAX;
        else if (y > 0 && a > 0 || y < 0 && a < C)
            FValue4Sort[nGlobalIndex] = FValues[nGlobalIndex];
        else
            FValue4Sort[nGlobalIndex] = -FLT_MAX + 1;
        Idx4Sort[nGlobalIndex] = nGlobalIndex;
    }
}

__device__ real devDiff;//fUp - fLow
__device__ real devRho;//bias
