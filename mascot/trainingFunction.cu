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

using std::cout;
using std::endl;

void trainingByGPU(vector<vector<float_point> > &v_v_DocVector, data_info &SDataInfo, SVMParam &param);

svmModel trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature) {

    vector<vector<float_point> > v_v_DocVector;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFile(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);
    svmProblem problem(v_v_DocVector, v_nLabel);
    svmModel model;
    param.probability = 0;//train with probability
    model.fit(problem, param);
    return model;
}

svm_model trainBinarySVM(svmProblem &problem, const SVMParam &param) {
    float_point pfCost = param.C;
    float_point pfGamma = param.gamma;
    CRBFKernel rbf(pfGamma);//ignore
    DeviceHessian ops(&rbf);

    CLATCache cacheStrategy((const int &) problem.getNumOfSamples());
    cout << "using " << cacheStrategy.GetStrategy() << endl;
    CSMOSolver s(&ops, &cacheStrategy);
    CSVMTrainer svmTrainer(&s);

    //compute Hessian Matrix
    string strHessianMatrixFileName = HESSIAN_FILE;
    string strDiagHessianFileName = HESSIAN_DIAG_FILE;

    //initialize Hessian IO operator

    int nNumofRowsOfHessianMatrix = (int) problem.getNumOfSamples();
    //space of row-index-in-file is for improving reading performace
    DeviceHessian::m_nNumofDim = (int) problem.getNumOfFeatures();
    DeviceHessian::m_nTotalNumofInstance = nNumofRowsOfHessianMatrix;

    //initial Hessian accessor
    SeqAccessor accessor;
    accessor.m_nTotalNumofInstance = DeviceHessian::m_nTotalNumofInstance;
    accessor.SetInvolveData(0, problem.getNumOfSamples() - 1, -1, -1);

    ops.SetAccessor(&accessor);

    //cache part of hessian matrix in memory
    timeval t1, t2;
    float_point elapsedTime;
    gettimeofday(&t1, NULL);
    gettimeofday(&t1, NULL);

    ops.PrecomputeKernelMatrix(problem.v_vSamples, &ops);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << elapsedTime << " ms.\n";

    /*FILE *pFile = fopen(strHessianMatrixFileName.c_str(), "rb");
    m_pTrainer->m_pSMOSolver->m_pHessianOps->ReadHessianFullRow(pFile, 0, nNumofHessianRow,  CHessianIOOps::m_pfHessianRowsInHostMem);
    fclose(pFile);*/

    gfNCost = pfCost;
    gfPCost = pfCost;
    gfGamma = pfGamma;
    ofstream writeOut(OUTPUT_FILE, ios::app | ios::out);
    writeOut << "Gamma=" << pfGamma << "; Cost=" << pfCost << endl;

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

    /* start n-fold-cross-validation */
    //allocate GPU memory for part of samples that are used to perform training.
    float_point *pfDevAlphaSubset;
    float_point *pfDevYiGValueSubset;
    int *pnDevLabelSubset;

    //get size of training samples
    int nNumofTrainingSamples = nTotalNumofSamples;

    //in n-fold-cross validation, the first (n -1) parts have the same size, so we can reuse memory
    checkCudaErrors(cudaMalloc((void **) &pfDevAlphaSubset, sizeof(float_point) * nNumofTrainingSamples));
    //checkCudaErrors(cudaMallocHost((void**)&pfDevYiGValueSubset, sizeof(float_point) * nNumofTrainingSamples));
    checkCudaErrors(cudaMalloc((void **) &pfDevYiGValueSubset, sizeof(float_point) * nNumofTrainingSamples));
    checkCudaErrors(cudaMalloc((void **) &pnDevLabelSubset, sizeof(int) * nNumofTrainingSamples));

    //set GPU memory
    checkCudaErrors(cudaMemset(pfDevAlphaSubset, 0, sizeof(float_point) * nNumofTrainingSamples));
    checkCudaErrors(cudaMemset(pfDevYiGValueSubset, -1, sizeof(float_point) * nNumofTrainingSamples));
    checkCudaErrors(cudaMemset(pnDevLabelSubset, 0, sizeof(int) * nNumofTrainingSamples));
    //copy training information to GPU for current training
    checkCudaErrors(cudaMemcpy(pfDevAlphaSubset, pfAlphaAll,
                               sizeof(float_point) * nTotalNumofSamples, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pfDevYiGValueSubset, pfYiGValueAll,
                               sizeof(float_point) * nTotalNumofSamples, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pnDevLabelSubset, pnLabelAll,
                               sizeof(int) * nTotalNumofSamples, cudaMemcpyHostToDevice));

    /************** train SVM model **************/
    //set data involved in training
    timeval tTraining1, tTraining2;
    float_point trainingElapsedTime;
    gettimeofday(&tTraining1, NULL);
    timespec timeTrainS, timeTrainE;
    clock_gettime(CLOCK_REALTIME, &timeTrainS);

    svm_model model;
    model.param.C = param.C;
    model.param.gamma = param.gamma;
    svmTrainer.SetInvolveTrainingData(0, nNumofTrainingSamples - 1, -1, -1);
    bool bTrain = svmTrainer.TrainModel(model, pfDevYiGValueSubset, pfDevAlphaSubset,
                                        pnDevLabelSubset, nNumofTrainingSamples, NULL);
    if (bTrain == false) {
        cerr << "can't find an optimal classifier" << endl;
    }
    if (ops.m_pKernelCalculater->GetType().compare(RBFKERNEL) == 0) {
        model.param.kernel_type = RBF;
    } else {
        cerr << "unsupported kernel type; Please contact the developers" << endl;
        exit(-1);
    }

    model.nDimension = problem.getNumOfFeatures();
    return model;
}

void evaluateSVMClassifier(svmModel &model, string strTrainingFileName, int nNumofFeature)
{ 
    vector<vector<float_point> > v_v_DocVector;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFile(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);

    //perform svm classification
    vector<int> predictLabels = model.predict(v_v_DocVector, true);
    int numOfCorrect = 0;
    for (int i = 0; i < v_v_DocVector.size(); ++i) 
    {
        if (predictLabels[i] == v_nLabel[i])
            numOfCorrect++;
//        for (int j = 0; j < problem.getNumOfClasses(); ++j) {
//            printf("%.2f,",prob[i][j]);
//        }
//        printf("\n");
    }
    printf("training accuracy = %.2f%%(%d/%d)\n", numOfCorrect / (float) v_v_DocVector.size()*100, 
            numOfCorrect, (int) v_v_DocVector.size());
}
