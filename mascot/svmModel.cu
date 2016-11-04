//
// Created by ss on 16-11-2.
//

#include <cstdio>
#include "svmModel.h"
#include "svmPredictor.h"
#include "../svm-shared/HessianIO/deviceHessian.h"
#include "../svm-shared/storageManager.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "trainingFunction.h"
unsigned int svmModel::getK(int i, int j) const {
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void svmModel::fit(const svmProblem &problem, const SVMParam &param) {
    nrClass = problem.getNumOfClasses();
    cnr2 = (nrClass) * (nrClass - 1) / 2;
    coef.clear();
    rho.clear();
    probA.clear();
    probB.clear();
    supportVectors.clear();
    label.clear();

    coef.resize(cnr2);
    rho.resize(cnr2);
    probA.resize(cnr2);
    probB.resize(cnr2);
    supportVectors.resize(cnr2);

    this->param = param;
    label = problem.label;
    for (int i = 0; i < nrClass; ++i) {
        for (int j = i + 1; j < nrClass; ++j) {
            svmProblem subProblem = problem.getSubProblem(i, j);
            printf("training classifier with label %d and %d\n", i, j);
            svm_model binaryModel = trainBinarySVM(subProblem, param);
            addBinaryModel(subProblem, binaryModel,i,j);
        }
    }
}

void svmModel::addBinaryModel(const svmProblem &problem, const svm_model &bModel, int i, int j) {
    unsigned int k = getK(i, j);
    for (int l = 0; l < bModel.nSV[0] + bModel.nSV[1]; ++l) {
//        printf("%d:%f.2|",bModel.pnIndexofSV[l],bModel.sv_coef[0][l]);
        coef[k].push_back(bModel.sv_coef[0][l]);
        supportVectors[k].push_back(problem.v_vSamples[bModel.pnIndexofSV[l]]);
//        printf("%d|",problem.v_nLabels[bModel.pnIndexofSV[l]]);
    }
//    printf("\n");
    rho[k] = bModel.rho[0];
}

vector<float_point*> svmModel::predictValues(const vector<vector<float_point> > &v_vSamples) const {
    vector<float_point *> decisionValues(cnr2);
    for (int k = 0; k < cnr2; ++k) {
            float_point *kernelValues = new float_point[v_vSamples.size() * supportVectors[k].size()];
            computeKernelValuesOnFly(v_vSamples, supportVectors[k], kernelValues);
            decisionValues[k] = predictLabels(kernelValues, (int) v_vSamples.size(), k);
            delete[] kernelValues;
        }
    return decisionValues;
}

vector<int> svmModel::predict(const vector<vector<float_point> > &v_vSamples) const {
    vector<float_point*> decisionValues = predictValues(v_vSamples);
    vector<int> labels;
    for (int l = 0; l < v_vSamples.size(); ++l) {
        vector<int> votes(nrClass,0);
        int k = 0;
        for (int i = 0; i < nrClass; ++i) {
            for (int j = i+1; j < nrClass; ++j) {
                if(decisionValues[k++][l]>0)
                    votes[i]++;
                else
                    votes[j]++;
            }
        }
        int maxVoteClass = 0;
        for (int i = 0; i < nrClass; ++i) {
            if (votes[i]>votes[maxVoteClass])
                maxVoteClass = i;
        }
        labels.push_back(this->label[maxVoteClass]);
    }
    return labels;
}

void
svmModel::computeKernelValuesOnFly(const vector<vector<float_point> > &samples,
                                   const vector<vector<float_point> > &supportVectors,
                                   float_point *kernelValues) const {
    for (int i = 0; i < samples.size(); ++i) {
        for (int j = 0; j < supportVectors.size(); ++j) {
            //rbf kernel
            float_point sum = 0;
            for (int k = 0; k < supportVectors[j].size(); ++k) {
                float_point d = samples[i][k] - supportVectors[j][k];
                sum += d * d;
            }
            kernelValues[i * supportVectors.size() + j] = (float_point) exp(-param.gamma * sum);
        }
    }
}

float_point *svmModel::predictLabels(const float_point *kernelValues, int nNumofTestSamples, int k) const {
    //get infomation from SVM model
    int nNumofSVs = (int) supportVectors[k].size();
//	int nNumofSVs = GetNumSV(pModel);
    float_point fBias = rho[k];
//	float_point fBias = *(pModel->rho);
    const float_point *pfSVsYiAlpha = coef[k].data();
//	float_point *pfSVsYiAlpha = pyfSVsYiAlpha[0];
//	int *pnSVsLabel = pModel->label;
//    float_point *pfYiAlphaofSVs;

    /*compute y_i*alpha_i*K(i, z) by GPU, where i is id of support vector.
     * pfDevSVYiAlphaHessian stores in the order of T1 sv1 sv2 ... T2 sv1 sv2 ... T3 sv1 sv2 ...
     */
    float_point *pfDevSVYiAlphaHessian;
    float_point *pfDevSVsYiAlpha;
//	int *pnDevSVsLabel;

    //if the memory is not enough for the storage when classifying all testing samples at once, divide it into multiple parts

    StorageManager *manager = StorageManager::getManager();
    long long int nMaxNumofFloatPoint = manager->GetFreeGPUMem();
    long long int nNumofPart = Ceil(nNumofSVs * nNumofTestSamples, nMaxNumofFloatPoint);

//	cout << "cache size is: " << nMaxNumofFloatPoint << " v.s.. " << nNumofSVs * nNumofTestSamples << endl;
//	cout << "perform classification in " << nNumofPart << " time(s)" << endl;

    //allocate memory for storing classification result
    float_point *pfClassificaitonResult = new float_point[nNumofTestSamples];
    //initialise the size of each part
    int *pSizeofPart = new int[nNumofPart];
    int nAverageSize = (int) (nNumofTestSamples / nNumofPart);
    for (int i = 0; i < nNumofPart; i++) {
        if (i != nNumofPart - 1) {
            pSizeofPart[i] = nAverageSize;
        } else {
            pSizeofPart[i] = nNumofTestSamples - nAverageSize * i;
        }
    }

    //perform classification for each part
    for (int i = 0; i < nNumofPart; i++) {
        checkCudaErrors(cudaMalloc((void **) &pfDevSVYiAlphaHessian, sizeof(float_point) * nNumofSVs * pSizeofPart[i]));
        checkCudaErrors(cudaMalloc((void **) &pfDevSVsYiAlpha, sizeof(float_point) * nNumofSVs));
//	checkCudaErrors(cudaMalloc((void**)&pnDevSVsLabel, sizeof(int) * nNumofSVs));

        checkCudaErrors(cudaMemset(pfDevSVYiAlphaHessian, 0, sizeof(float_point) * nNumofSVs * pSizeofPart[i]));
        checkCudaErrors(cudaMemset(pfDevSVsYiAlpha, 0, sizeof(float_point) * nNumofSVs));
//	checkCudaErrors(cudaMemset(pnDevSVsLabel, 0, sizeof(int) * nNumofSVs));

        checkCudaErrors(cudaMemcpy(pfDevSVYiAlphaHessian, kernelValues + i * nAverageSize * nNumofSVs,
                                   sizeof(float_point) * nNumofSVs * pSizeofPart[i], cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy(pfDevSVsYiAlpha, pfSVsYiAlpha, sizeof(float_point) * nNumofSVs, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(pnDevSVsLabel, pnSVsLabel, sizeof(int) * nNumofSVs, cudaMemcpyHostToDevice));

        //compute y_i*alpha_i*K(i, z)
        int nVecMatxMulGridDimY = pSizeofPart[i];
        int nVecMatxMulGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
        dim3 vecMatxMulGridDim(nVecMatxMulGridDimX, nVecMatxMulGridDimY);
        VectorMatrixMul << < vecMatxMulGridDim, BLOCK_SIZE >> >
                                                (pfDevSVsYiAlpha, pfDevSVYiAlphaHessian, pSizeofPart[i], nNumofSVs);

        //perform classification
        ComputeClassLabel(pSizeofPart[i], pfDevSVYiAlphaHessian,
                          nNumofSVs, fBias, pfClassificaitonResult + i * nAverageSize);

        if (pfClassificaitonResult == NULL) {
            cerr << "error in ComputeClassLabel" << endl;
            exit(-1);
        }


        //free memory
        checkCudaErrors(cudaFree(pfDevSVYiAlphaHessian));
        pfDevSVYiAlphaHessian = NULL;
        checkCudaErrors(cudaFree(pfDevSVsYiAlpha));
//	checkCudaErrors(cudaFree(pnDevSVsLabel));
    }

    return pfClassificaitonResult;
}

/*
 * @brief: compute/predict the labels of testing samples
 * @output: a set of class labels, associated to testing samples
 */
float_point *svmModel::ComputeClassLabel(int nNumofTestSamples,
                                         float_point *pfDevSVYiAlphaHessian, const int &nNumofSVs,
                                         float_point fBias, float_point *pfFinalResult) const {
    float_point *pfReturn = NULL;
    if (nNumofTestSamples <= 0 ||
        pfDevSVYiAlphaHessian == NULL ||
        nNumofSVs <= 0) {
        cerr << "error in ComputeClassLabel: invalid input params" << endl;
        return pfReturn;
    }

    //compute the size of current processing testing samples
/*	size_t nFreeMemory,nTotalMemory;
	cudaMemGetInfo(&nFreeMemory,&nTotalMemory);
*/    int nMaxSizeofProcessingSample = ((CACHE_SIZE) * 1024 * 1024 * 4 / (sizeof(float_point) * nNumofSVs));

    //reduce by half
    nMaxSizeofProcessingSample = nMaxSizeofProcessingSample / 2;

    //if the number of samples in small
    if (nMaxSizeofProcessingSample > nNumofTestSamples) {
        nMaxSizeofProcessingSample = nNumofTestSamples;
    }
    //compute grid size, and block size for partial sum
    int nPartialGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
    int nPartialGridDimY = nMaxSizeofProcessingSample;
    dim3 dimPartialSumGrid(nPartialGridDimX, nPartialGridDimY);
    dim3 dimPartialSumBlock(BLOCK_SIZE);

    //compute grid size, and block size for global sum and class label computing
    int nGlobalGridDimX = 1;
    int nGlobalGridDimY = nMaxSizeofProcessingSample;
    dim3 dimGlobalSumGrid(nGlobalGridDimX, nGlobalGridDimY); //can use 1D grid
    dim3 dimGlobalSumBlock(nPartialGridDimX);

    //memory for computing partial sum by GPU
    float_point *pfDevPartialSum;
//	cout << "dimx=" << nPartialGridDimX << "; dimy=" << nPartialGridDimY << endl;
    checkCudaErrors(cudaMalloc((void **) &pfDevPartialSum, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));
    checkCudaErrors(cudaMemset(pfDevPartialSum, 0, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));

    //memory for computing global sum by GPU
    float_point *pfDevClassificationResult;
    checkCudaErrors(cudaMalloc((void **) &pfDevClassificationResult, sizeof(float_point) * nGlobalGridDimY));
    checkCudaErrors(cudaMemset(pfDevClassificationResult, 0, sizeof(float_point) * nGlobalGridDimY));

    //reduce step size of partial sum, and global sum
    int nPartialReduceStepSize = 0;
    nPartialReduceStepSize = (int) pow(2, (ceil(log2((float) BLOCK_SIZE)) - 1));
    int nGlobalReduceStepSize = 0;
    nGlobalReduceStepSize = (int) pow(2, ceil(log2((float) nPartialGridDimX)) - 1);

    for (int nStartPosofTestSample = 0;
         nStartPosofTestSample < nNumofTestSamples; nStartPosofTestSample += nMaxSizeofProcessingSample) {
        if (nStartPosofTestSample + nMaxSizeofProcessingSample > nNumofTestSamples) {
            //the last part of the testing samples
            nMaxSizeofProcessingSample = nNumofTestSamples - nStartPosofTestSample;
            nPartialGridDimY = nMaxSizeofProcessingSample;
            dimPartialSumGrid = dim3(nPartialGridDimX, nPartialGridDimY);
            nGlobalGridDimY = nMaxSizeofProcessingSample;
            dimGlobalSumGrid = dim3(nGlobalGridDimX, nGlobalGridDimY);

            checkCudaErrors(cudaFree(pfDevPartialSum));
            checkCudaErrors(
                    cudaMalloc((void **) &pfDevPartialSum, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));
            checkCudaErrors(cudaMemset(pfDevPartialSum, 0, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));

            checkCudaErrors(cudaFree(pfDevClassificationResult));
            checkCudaErrors(cudaMalloc((void **) &pfDevClassificationResult, sizeof(float_point) * nGlobalGridDimY));
            checkCudaErrors(cudaMemset(pfDevClassificationResult, 0, sizeof(float_point) * nGlobalGridDimY));
        }
        /********* compute partial sum **********/
        ComputeKernelPartialSum << < dimPartialSumGrid, dimPartialSumBlock, BLOCK_SIZE * sizeof(float_point) >> >
                                                                            (pfDevSVYiAlphaHessian, nNumofSVs, pfDevPartialSum,
                                                                                    nPartialReduceStepSize);
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cerr << "cuda error in ComputeClassLabel: failed at ComputePartialSum: " << cudaGetErrorString(error)
                 << endl;
            return pfReturn;
        }

        /********** compute global sum and class label *********/
        //compute global sum
        ComputeKernelGlobalSum << < dimGlobalSumGrid, dimGlobalSumBlock, nPartialGridDimX * sizeof(float_point) >> >
                                                                         (pfDevClassificationResult, fBias,
                                                                                 pfDevPartialSum, nGlobalReduceStepSize);
        cudaDeviceSynchronize();

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            cerr << "cuda error in ComputeClassLabel: failed at ComputeGlobalSum: " << cudaGetErrorString(error)
                 << endl;
            return pfReturn;
        }

        //copy classification result back
        checkCudaErrors(cudaMemcpy(pfFinalResult + nStartPosofTestSample, pfDevClassificationResult,
                                   nMaxSizeofProcessingSample * sizeof(float_point), cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaFree(pfDevPartialSum));
    checkCudaErrors(cudaFree(pfDevClassificationResult));

    pfReturn = pfFinalResult;
    return pfReturn;
}

