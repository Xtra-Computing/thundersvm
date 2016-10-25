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
#include <helper_cuda.h>
using std::cout;
using std::endl;

void trainingByGPU(vector<vector<float_point> > &v_v_DocVector, data_info &SDataInfo, SVMParam &param);

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature)
{
	//initialize grid
	data_info SDataInfo;
	//SDataInfo.nNumofSample = NUMOFSAMPLE;//#######
	//string strTrainingFileName = "dataset/gisette_scale";

	vector<vector<float_point> > v_v_DocVector;
	vector<int> v_nLabel;

	CDataIOOps rawDataRead;
	rawDataRead.ReadFromFile(strTrainingFileName, nNumofFeature, v_v_DocVector, v_nLabel);
	//cout << v_v_DocVector.size() << " : " << v_nLabel.size() << endl;

	SDataInfo.nNumofExample = v_v_DocVector.size();
	//get labels of data
	int nSizeofSample = v_nLabel.size();
	SDataInfo.pnLabel = new float[nSizeofSample];

	for(int l = 0; l < nSizeofSample; l++)
	{
		if(v_nLabel[l] != 1 && v_nLabel[l] != -1)
		{
			cerr << "error label" << endl;
			exit(0);
		}
		SDataInfo.pnLabel[l] = v_nLabel[l];
	}

	trainingByGPU(v_v_DocVector, SDataInfo, param);
	delete[] SDataInfo.pnLabel;
}

void trainingByGPU(vector<vector<float_point> > &v_v_DocVector, data_info &SDataInfo, SVMParam &param)
{
	float_point pfCost = param.C;
	float_point pfGamma = param.gamma;
	CRBFKernel rbf(pfGamma);//ignore
	DeviceHessian ops(&rbf);

	CLATCache cacheStrategy(SDataInfo.nNumofExample);
	cout << "using " << cacheStrategy.GetStrategy() << endl;
	CSMOSolver s(&ops, &cacheStrategy);
	CSVMTrainer svmTrainer(&s);
	int nNumofSample = SDataInfo.nNumofExample;

	//compute Hessian Matrix
	string strHessianMatrixFileName = HESSIAN_FILE;
	string strDiagHessianFileName = HESSIAN_DIAG_FILE;

	//initialize Hessian IO operator

	int nNumofRowsOfHessianMatrix = v_v_DocVector.size();
	//space of row-index-in-file is for improving reading performace
	DeviceHessian::m_nNumofDim = v_v_DocVector.front().size();
	DeviceHessian::m_nTotalNumofInstance = nNumofRowsOfHessianMatrix;

	//initial Hessian accessor
	SeqAccessor accessor;
	accessor.m_nTotalNumofInstance = DeviceHessian::m_nTotalNumofInstance;
	accessor.SetInvolveData(0, SDataInfo.nNumofExample - 1, -1, -1);

	ops.SetAccessor(&accessor);

	//cache part of hessian matrix in memory
	struct sysinfo info;
	sysinfo(&info);
	long nFreeMemInFloat = (info.freeram / sizeof(float_point));
	//memory for storing sample data, both original and transposed forms. That's why we use "2" here.
	long nMemForSamples = (ops.m_nNumofDim * ops.m_nTotalNumofInstance * 2);
	nFreeMemInFloat -= nMemForSamples; //get the number of available memory in the form of number of float
	nFreeMemInFloat *= 0.9; //use 80% of the memory for caching
	long nNumofHessianRow = (nFreeMemInFloat / nNumofSample);
	assert(nFreeMemInFloat > 0);
	if (nNumofHessianRow > nNumofSample)
	{
		//if the available memory is available to store the whole hessian matrix
		nNumofHessianRow = nNumofSample;
	}
	//			if(nNumofHessianRow > 21500)nNumofHessianRow = 21500;
	//			assert(nNumofHessianRow == 21500);
/*	long nRAMForRow = RAM_SIZE * 1024;
	nRAMForRow *= 1024;
	nRAMForRow *= 1024;
	nRAMForRow /= sizeof(float_point);
	nNumofHessianRow = (nRAMForRow / nNumofSample);
	if(nNumofHessianRow > nNumofSample)
		nNumofHessianRow = nNumofSample;
*/	cout << nNumofHessianRow << " rows cached in RAM" << endl;
	long lSizeofCachedHessia = sizeof(float_point) * nNumofHessianRow * nNumofSample;
	checkCudaErrors(cudaMallocHost((void**)&DeviceHessian::m_pfHessianRowsInHostMem,
					sizeof(float_point) * nNumofHessianRow * nNumofSample));
	memset(DeviceHessian::m_pfHessianRowsInHostMem, 0, lSizeofCachedHessia);
	DeviceHessian::m_nNumofCachedHessianRow = nNumofHessianRow;
	DeviceHessian::m_pfHessianDiag = new float_point[ops.m_nTotalNumofInstance];
//	ops.m_pfHessianDiagTest = new float_point[ops.m_nTotalNumofInstance];

	//pre-compute Hessian Matrix and store the result into a file
	timeval t1, t2;
	float_point elapsedTime;
	gettimeofday(&t1, NULL);
	gettimeofday(&t1, NULL);
	bool bWriteHessian = ops.PrecomputeHessian(strHessianMatrixFileName, strDiagHessianFileName, v_v_DocVector);
	ops.ReadDiagFromHessianMatrix();

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
	float *pnLabelAll = SDataInfo.pnLabel;
	int nTotalNumofSamples = SDataInfo.nNumofExample;

	/* allocate GPU device memory */
	//set default value at
	float_point *pfAlphaAll = new float_point[nTotalNumofSamples];
	float_point *pfYiGValueAll = new float_point[nTotalNumofSamples];
	for (int i = 0; i < nTotalNumofSamples; i++)
	{
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
	checkCudaErrors(cudaMalloc((void**)&pfDevAlphaSubset, sizeof(float_point) * nNumofTrainingSamples));
	//checkCudaErrors(cudaMallocHost((void**)&pfDevYiGValueSubset, sizeof(float_point) * nNumofTrainingSamples));
	checkCudaErrors(cudaMalloc((void**)&pfDevYiGValueSubset, sizeof(float_point) * nNumofTrainingSamples));
	checkCudaErrors(cudaMalloc((void**)&pnDevLabelSubset, sizeof(int) * nNumofTrainingSamples));

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
	svmTrainer.SetInvolveTrainingData(0, nNumofTrainingSamples - 1, -1, -1);
	bool bTrain = svmTrainer.TrainModel(model, pfDevYiGValueSubset, pfDevAlphaSubset,
										pnDevLabelSubset, nNumofTrainingSamples, NULL);
	if (bTrain == false)
	{
		cerr << "can't find an optimal classifier" << endl;
	}
	if(ops.m_pKernelCalculater.GetType().compare(RBFKERNEL) == 0)
	{
		model.param.kernel_type = RBF;
	}
	else
	{
		cerr << "unsupported kernel type; Please contact the developers" << endl;
		exit(-1);
	}

	model.nDimension = v_v_DocVector[0].size();
	svmTrainer.SaveModel("svm.model", &model, v_v_DocVector);
	cout << " rho = " << model.rho[0] << "; # of SVs = " << model.nSV[0] + model.nSV[1] << endl;
	gettimeofday(&tTraining2, NULL);
	clock_gettime(CLOCK_REALTIME, &timeTrainE);
	long lTrainingTime = ((timeTrainE.tv_sec - timeTrainS.tv_sec) * 1e9 + (timeTrainE.tv_nsec - timeTrainS.tv_nsec));

	trainingElapsedTime = (tTraining2.tv_sec - tTraining1.tv_sec) * 1000.0;
	trainingElapsedTime += (tTraining2.tv_usec - tTraining1.tv_usec) / 1000.0;
	cout << "training time: " << trainingElapsedTime << " ms v.s. "
			<< lTrainingTime / 1000000 << " ms" << endl;

	cout << "updating alpha: " << nTimeOfUpdateAlpha / 1000 << " ms." << endl;
	nTimeOfUpdateAlpha = 0;
	cout << "select 1st: " << nTimeOfSelect1stSample / 1000 << " ms." << endl;
	nTimeOfSelect1stSample = 0;
	cout << "select 2nd: " << nTimeOfSelect2ndSample / 1000 << " ms." << endl;
	nTimeOfSelect2ndSample = 0;
	cout << "updating YiF: " << nTimeOfUpdateYiFValue / 1000 << " ms." << endl;
	nTimeOfUpdateYiFValue = 0;
	cout << "loop: " << nTimeOfLoop / 1000000 << " ms." << endl;
	nTimeOfLoop = 0;
	cout << "preparation: " << nTimeOfPrep / 1000000 << " ms." << endl;
	nTimeOfPrep = 0;
	cout << "IO timer: " << lIO_timer / 1000000 << " ms" << endl;

	cout << "GetHessian timer: " << lGetHessianRowTime / 1000000 << " ms"
			<< endl;
	cout << "IO counter: " << lIO_counter << " v.s GetHessianRow counter: "
			<< lGetHessianRowCounter << endl;
	lIO_counter = 0;
	lGetHessianRowCounter = 0;

	cout << "Ram " << lRamHitCount << "; SSD " << lSSDHitCount << endl;
	lRamHitCount = 0;
	lSSDHitCount = 0;

	cout << "get: " << lCountNormal << "; latest: " << lCountLatest << endl;
	lCountNormal = 0;
	lCountLatest = 0;

	checkCudaErrors(cudaFree(pfDevAlphaSubset));
	checkCudaErrors(cudaFree(pnDevLabelSubset));
	checkCudaErrors(cudaFree(pfDevYiGValueSubset));
	//checkCudaErrors(cudaFreeHost(pfDevYiGValueSubset));

	delete[] pfAlphaAll;
	delete[] pfYiGValueAll;

	//release pinned memory
	cudaFreeHost(DeviceHessian::m_pfHessianRowsInHostMem);
	delete[] DeviceHessian::m_pfHessianDiag;
//	delete[] ops.m_pfHessianDiagTest;

	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
	cout << elapsedTime << " ms.\n";
}
