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
#include <cuda_profiler_api.h>

#include "svmProblem.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"

using std::cout;
using std::endl;

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature, SvmModel &model) {

    vector<vector<svm_node> > v_v_DocVector;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFileSparse(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);
    SvmProblem problem(v_v_DocVector, v_nLabel);
    model.fit(problem, param);
}

svm_model trainBinarySVM(SvmProblem &problem, const SVMParam &param) {
    float_point pfCost = param.C;
    float_point pfGamma = param.gamma;
//    RBFKernelFunction f = RBFKernelFunction(param.gamma);
    DeviceHessianOnFly ops(problem,param.gamma);

    CLATCache cacheStrategy((const int &) problem.getNumOfSamples());
    cout << "using " << cacheStrategy.GetStrategy() << endl;
    CSMOSolver s(&ops, &cacheStrategy);
    CSVMTrainer svmTrainer(&s);

    gfNCost = pfCost;
    gfPCost = pfCost;

    //copy training information from input parameters
    const int *pnLabelAll = problem.v_nLabels.data();
    int nTotalNumofSamples = (int) problem.getNumOfSamples();

    /* allocate GPU device memory */
    //set default value at
    float_point *pfAlphaAll = new float_point[nTotalNumofSamples];
    float_point *pfYiGValueAll = new float_point[nTotalNumofSamples];
    for (int i = 0; i < nTotalNumofSamples; i++) {
        //initially, the values of alphas are 0s
        pfAlphaAll[i] = 0;
        //GValue is -y_i, as all alphas are 0s. YiGValue is always -1
        pfYiGValueAll[i] = -pnLabelAll[i];
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
    checkCudaErrors(cudaMemcpy(pfDevAlphaSubset, pfAlphaAll,
                               sizeof(float_point) * nTotalNumofSamples, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pfDevYiGValueSubset, pfYiGValueAll,
                               sizeof(float_point) * nTotalNumofSamples, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pnDevLabelSubset, pnLabelAll,
                               sizeof(int) * nTotalNumofSamples, cudaMemcpyHostToDevice));

    /************** train SVM model **************/
    svm_model model;
    model.param.C = param.C;
    model.param.gamma = param.gamma;
    svmTrainer.SetInvolveTrainingData(0, nNumofTrainingSamples - 1, -1, -1);
    bool bTrain = svmTrainer.TrainModel(model, pfDevYiGValueSubset, pfDevAlphaSubset,
                                        pnDevLabelSubset, nNumofTrainingSamples, NULL);
    if (bTrain == false) {
        cerr << "can't find an optimal classifier" << endl;
    }


    //free device memory
    checkCudaErrors(cudaFree(pfDevAlphaSubset));
    checkCudaErrors(cudaFree(pnDevLabelSubset));
    checkCudaErrors(cudaFree(pfDevYiGValueSubset));

    delete[] pfAlphaAll;
    delete[] pfYiGValueAll;

    return model;
}

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature)
{
    vector<vector<svm_node> > v_v_DocVector;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFileSparse(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);

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
