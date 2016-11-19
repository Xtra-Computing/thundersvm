/*
 * testTrainer.cpp
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi
 */

#include "trainingFunction.h"

#include<iostream>
#include<cassert>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/sysinfo.h>

#include "../svm-shared/gpu_global_utility.h"
#include "../svm-shared/constant.h"
#include "../svm-shared/HessianIO/baseHessian.h"
#include "../svm-shared/HessianIO/seqAccessor.h"
#include "../svm-shared/kernelCalculater/kernelCalculater.h"
#include "../svm-shared/svmTrainer.h"
#include "../svm-shared/smoSolver.h"
#include "../svm-shared/Cache/cache.h"
#include "DataIOOps/DataIO.h"
#include "DataIOOps/BaseLibsvmReader.h"
#include <helper_cuda.h>

#include "svmProblem.h"
#include "../svm-shared/HessianIO/hostHessianOnFly.h"

using std::cout;
using std::endl;

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature, SvmModel &model) {

    vector<vector<float_point> > v_v_DocVector;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFile(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);
    SvmProblem problem(v_v_DocVector, v_nLabel);
    cout << "Training data loaded. Start training..." << endl;
    model.fit(problem, param);
}

svm_model trainBinarySVM(SvmProblem &problem, const SVMParam &param, cudaStream_t stream) {
    float_point pfCost = param.C;
    float_point pfGamma = param.gamma;
    RBFKernelFunction f = RBFKernelFunction(param.gamma);
    HostHessianOnFly ops(f,problem.v_vSamples);

    CLATCache cacheStrategy((const int &) problem.getNumOfSamples());
    cout << "using " << cacheStrategy.GetStrategy() << endl;
    CSMOSolver s(&ops, &cacheStrategy);
    CSVMTrainer svmTrainer(&s);
    svmTrainer.setStream(stream);

    gfNCost = pfCost;
    gfPCost = pfCost;

    //copy training information from input parameters
//    const int *pnLabelAll = problem.v_nLabels.data();
    int nTotalNumofSamples = (int) problem.getNumOfSamples();

    /* allocate GPU device memory */
    //set default value at
//    float_point *pfAlphaAll = new float_point[nTotalNumofSamples];
//    float_point *pfYiGValueAll = new float_point[nTotalNumofSamples];
    int *pnLabelAll;
    float_point *pfAlphaAll;
    float_point *pfYiGValueAll;
    checkCudaErrors(cudaMallocHost((void **)&pnLabelAll,sizeof(int)*nTotalNumofSamples));
    checkCudaErrors(cudaMallocHost((void **)&pfAlphaAll,sizeof(float_point)*nTotalNumofSamples));
    checkCudaErrors(cudaMallocHost((void **)&pfYiGValueAll,sizeof(float_point)*nTotalNumofSamples));
    for (int i = 0; i < nTotalNumofSamples; i++) {
        //initially, the values of alphas are 0s
        pfAlphaAll[i] = 0;
        //GValue is -y_i, as all alphas are 0s. YiGValue is always -1
//        pfYiGValueAll[i] = -pnLabelAll[i];
        pfYiGValueAll[i] = -problem.v_nLabels[i];
        pnLabelAll[i] = problem.v_nLabels[i];
    }

    //allocate GPU memory for part of samples that are used to perform training.
    float_point *pfDevAlphaSubset;
    float_point *pfDevYiGValueSubset;
    int *pnDevLabelSubset;

    //get size of training samples
    int nNumofTrainingSamples = nTotalNumofSamples;

    checkCudaErrors(cudaMalloc((void **) &pfDevAlphaSubset, sizeof(float_point) * nNumofTrainingSamples));
    checkCudaErrors(cudaMalloc((void **) &pfDevYiGValueSubset, sizeof(float_point) * nNumofTrainingSamples));
    checkCudaErrors(cudaMalloc((void **) &pnDevLabelSubset, sizeof(int) * nNumofTrainingSamples));

    //copy training information to GPU for current training
    checkCudaErrors(cudaMemcpyAsync(pfDevAlphaSubset, pfAlphaAll,
                               sizeof(float_point) * nTotalNumofSamples, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(pfDevYiGValueSubset, pfYiGValueAll,
                               sizeof(float_point) * nTotalNumofSamples, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(pnDevLabelSubset, pnLabelAll,
                               sizeof(int) * nTotalNumofSamples, cudaMemcpyHostToDevice, stream));
//    checkCudaErrors(cudaStreamSynchronize(stream));

    /************** train SVM model **************/
    svm_model model;
    model.param.C = param.C;
    model.param.gamma = param.gamma;
    svmTrainer.SetInvolveTrainingData(0, nNumofTrainingSamples - 1, -1, -1);
    svmTrainer.setStream(stream);
    cout << "completed data and model preparation. Start training SVM model" << endl;
    bool bTrain = svmTrainer.TrainModel(model, pfDevYiGValueSubset, pfDevAlphaSubset,
                                        pnDevLabelSubset, nNumofTrainingSamples, NULL);
    if (bTrain == false) {
        cerr << "can't find an optimal classifier" << endl;
    }

    model.nDimension = problem.getNumOfFeatures();

    //free device memory
    checkCudaErrors(cudaFree(pfDevAlphaSubset));
    checkCudaErrors(cudaFree(pnDevLabelSubset));
    checkCudaErrors(cudaFree(pfDevYiGValueSubset));

    checkCudaErrors(cudaFreeHost(pnLabelAll));
    checkCudaErrors(cudaFreeHost(pfAlphaAll));
    checkCudaErrors(cudaFreeHost(pfYiGValueAll));
//    delete[] pfAlphaAll;
//    delete[] pfYiGValueAll;


    return model;
}

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature)
{ 
    vector<vector<float_point> > v_v_DocVector;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFile(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);

    //perform svm classification
    vector<int> predictLabels = model.predict(v_v_DocVector, model.isProbability());
    int numOfCorrect = 0;
    for (int i = 0; i < v_v_DocVector.size(); ++i) 
    {
        if (predictLabels[i] == v_nLabel[i])
            numOfCorrect++;
    }
    printf("training accuracy = %.2f%%(%d/%d)\n", numOfCorrect / (float) v_v_DocVector.size()*100, 
            numOfCorrect, (int) v_v_DocVector.size());
}
