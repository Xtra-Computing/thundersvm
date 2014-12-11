/*
 * @brief: this file contains the definition of svm trainer class
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 */

#include "svmTrainer.h"
#include "time.h"
#include "gpu_global_utility.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <sys/time.h>

long nTimeOfLoop = 0;
long nTimeOfPrep = 0;

/*
 * @brief: train svm model. Training data is consisted of two parts for n-fold cross validation scenario;
 * @param: model: the svm model of the training
 * @param: pfDevYiFValueSubset: gradient of sub set of samples (training samples are sub set of the whole samples)
 */
int nInterater = 0;
bool CSVMTrainer::TrainModel(svm_model &model, float_point *pfDevYiFValueSubset,
							 float_point *pfDevAlphaSubset, int *pnDevLabelSubset,
							 int nNumofTrainingSamples)
{
	timespec timePrepS, timePrepE;
	clock_gettime(CLOCK_REALTIME, &timePrepS);

	bool bReturn = true;

	assert(nNumofTrainingSamples > 0  && pfDevAlphaSubset != NULL &&
		   pfDevYiFValueSubset != NULL && pnDevLabelSubset != NULL);

	/*************** prepare to perform training *************/
	//allocate memory for SMO slover
	bool bPreperation = m_pSMOSolver->SMOSolverPreparation(nNumofTrainingSamples);
	assert(bPreperation != false);

	//initialize cache
	long nMaxNumofFloatPoint = (long(CACHE_SIZE) * 1024 * 1024 / 4) - (3 * nNumofTrainingSamples);//GetFreeGPUMem(); use 500MB memory
	//GPU memory stores a Hessian diagonal, and a few rows of Hessian matrix
	int nMaxCacheSize = (nMaxNumofFloatPoint - nNumofTrainingSamples) / nNumofTrainingSamples;
	//GPU memory can't go to 100%, so we use a ratio here
	int nSizeofCache = nMaxCacheSize;// * 0.95;
//	if(nSizeofCache > 2627)nSizeofCache = 2627;
//nSizeofCache = DIMENSION;
	if(nSizeofCache > nNumofTrainingSamples)
	{
		nSizeofCache = nNumofTrainingSamples;
	}

/*	cout << "cache size is: " << nSizeofCache << " max: " << nMaxCacheSize << "; max num of float_point: "
			<< nMaxNumofFloatPoint << "; percentage of cached samples: " <<
			((float_point)100 * nSizeofCache) / nNumofTrainingSamples << "%" << endl;
*/

	timeval tInit1, tInit2;
	float_point InitCacheElapsedTime;
	gettimeofday(&tInit1, NULL);
	//initialize cache
	bool bInitCache = m_pSMOSolver->InitCache(nSizeofCache, nNumofTrainingSamples);
	assert(bInitCache != false);
//	cout << "cache initialized" << endl;
	gettimeofday(&tInit2, NULL);
	InitCacheElapsedTime = (tInit2.tv_sec - tInit1.tv_sec) * 1000.0;
	InitCacheElapsedTime += (tInit2.tv_usec - tInit1.tv_usec) / 1000.0;
//	cout << "initCache time: " << InitCacheElapsedTime << " ms.\n";

	clock_gettime(CLOCK_REALTIME, &timePrepE);
	long lTempPrep = ((timePrepE.tv_sec - timePrepS.tv_sec) * 1e9 + (timePrepE.tv_nsec - timePrepS.tv_nsec));
	if(lTempPrep > 0)
		nTimeOfPrep += lTempPrep;
	else
		cout << "preparation timer error" << endl;

	//start training process
	int nIter = 0;
	int nMaxIter = (nNumofTrainingSamples > INT_MAX / ITERATION_FACTOR ? INT_MAX : ITERATION_FACTOR * nNumofTrainingSamples) * 4;
	int nSelectFirstSample = -1;
	int nSelectSecondeSample = -1;

	timespec timeLoopS, timeLoopE;
	clock_gettime(CLOCK_REALTIME, &timeLoopS);
	//allocate memory for reading hessian row
	m_pSMOSolver->m_pHessianReader->AllocateBuffer(1);

	cudaStreamCreate(&m_pSMOSolver->m_stream1_Hessian_row);//for overlapping memcpy
	m_pSMOSolver->m_pfDevGValue = pfDevYiFValueSubset;
	checkCudaErrors(cudaMemcpy(m_pSMOSolver->m_pnLabel, pnDevLabelSubset, sizeof(int) * nNumofTrainingSamples, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_pSMOSolver->m_pfGValue, pfDevYiFValueSubset, sizeof(float_point) * nNumofTrainingSamples, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_pSMOSolver->m_pfAlpha, pfDevAlphaSubset, sizeof(float_point) * nNumofTrainingSamples, cudaMemcpyDeviceToHost));

/*	ofstream out, access_count;
	out.open("access.txt", ios::out | ios::app);
	access_count.open("access_count.txt", ios::out | ios::app);
	int nPreOne = 0, nPreTwo = 0;
	int nValid = 0;
	int nInValid = 0;
	set<int> used_set1;
	set<int> used_set2;
	int count[200000] = {0};
	int *nLastAccess;
	nLastAccess = new int[nNumofTrainingSamples];
	memset(nLastAccess, 0, sizeof(int) * nNumofTrainingSamples);
	int *numofAccess = new int[nNumofTrainingSamples];
	memset(numofAccess, 0, sizeof(int) * nNumofTrainingSamples);
*/
	//cudaProfilerStart();
	while(nIter < nMaxIter)
	{
		int nEnd = m_pSMOSolver->Iterate(pfDevYiFValueSubset, pfDevAlphaSubset, pnDevLabelSubset, nNumofTrainingSamples);

		if(nEnd == 1)
		{
			cout << " Done" << endl;
			break;
		}
		if(nIter % 1000 == 0 && nIter != 0)
		{
			cout << ".";
			cout.flush();
		}
		nIter++;

	}

	cudaStreamDestroy(m_pSMOSolver->m_stream1_Hessian_row);//destroy stream
	//release buffer for reading hessian row
	m_pSMOSolver->m_pHessianReader->ReleaseBuffer();

	//m_pSMOSolver->m_pCache->PrintCache();

	clock_gettime(CLOCK_REALTIME, &timeLoopE);
	long lTempLoop = ((timeLoopE.tv_sec - timeLoopS.tv_sec) * 1e9 + (timeLoopE.tv_nsec - timeLoopS.tv_nsec));
	if(lTempLoop > 0)
		nTimeOfLoop += lTempLoop;
	else
		cout << "loop timer error" << endl;

	cout << "# of interation: " << nIter << endl;

	//store classification result in SVM Model
	int *pnLabel = new int[nNumofTrainingSamples];
	float_point *pfAlpha = new float_point[nNumofTrainingSamples];
	float_point *pfYiFValue = new float_point[nNumofTrainingSamples];
	memset(pnLabel, 0, sizeof(int) * nNumofTrainingSamples);
	memset(pfAlpha, 0, sizeof(float_point) * nNumofTrainingSamples);
	memset(pfYiFValue, 0, sizeof(float_point) * nNumofTrainingSamples);

	checkCudaErrors(cudaMemcpy(pnLabel, pnDevLabelSubset, sizeof(int) * nNumofTrainingSamples, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pfAlpha, pfDevAlphaSubset, sizeof(float_point) * nNumofTrainingSamples, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pfYiFValue, pfDevYiFValueSubset, sizeof(float_point) * nNumofTrainingSamples, cudaMemcpyDeviceToHost));

	//compute the # of support vectors, and get bias of the model
	float_point *pfYiAlphaTemp = new float_point[nNumofTrainingSamples];
	float_point *pfPositiveAlphaTemp = new float_point[nNumofTrainingSamples];
	float_point *pfNegativeAlphaTemp = new float_point[nNumofTrainingSamples];
	int *pnIndexofSVTemp = new int[nNumofTrainingSamples];

	float_point fYGsumofFreeAlpha = 0;
	int nNumofFreeAlpha = 0;
	int nNumofSVs = 0;
	int nNumofPSV = 0, nNumofNSV = 0;

	for(int i = 0; i < nNumofTrainingSamples; i++)
	{
		/*if(pfAlpha[i] == 0)
		{
			cout << "alpha=0: " << m_pSMOSolver->m_pfGValue[i] << endl;
			if(i % 1000 == 0)sleep(1);
		}
		if(pfAlpha[i] == gfPCost)
		{
			cout << "alpha=c: " << m_pSMOSolver->m_pfGValue[i] << endl;
			if(i % 1000 == 0)sleep(1);
		}*/
		if(pfAlpha[i] > 0)
		{
			//keep support vector in temp memory
			pfYiAlphaTemp[nNumofSVs] = pnLabel[i] * pfAlpha[i];
			pnIndexofSVTemp[nNumofSVs] = i;
			//convert index to global index in Hessian matrix
			m_pSMOSolver->MapIndexToHessian(pnIndexofSVTemp[nNumofSVs]);
			//count the # of support vectors
			nNumofSVs++;
			if(pnLabel[i] > 0)
			{
				pfPositiveAlphaTemp[nNumofPSV] = pfAlpha[i];
				nNumofPSV++;
			}
			else
			{
				pfNegativeAlphaTemp[nNumofNSV] = pfAlpha[i];
				nNumofNSV++;
			}

			//for getting bias
			if((pfAlpha[i] < gfPCost && pnLabel[i] > 0) || (pfAlpha[i] < gfNCost && pnLabel[i] < 0))
			{
				fYGsumofFreeAlpha += pfYiFValue[i];//(pnLabel[i] * pfYiFValue[i]);
				nNumofFreeAlpha++;
			}
		}
		if(pfAlpha[i] < 0)
		{
			cerr << "potential problem in TrainModel: alpha < 0" << endl;
		}
	}

	cout << "# free SV " << nNumofFreeAlpha << "; # of SVs = " << nNumofSVs << endl;
	/**************** store result to SVM model ***********************/
	model.label = new int[2];
	model.nSV = new int[2];
	model.label[0] = 1;
	model.label[1] = -1;
	model.nSV[0] = nNumofPSV;
	model.nSV[1] = nNumofNSV;

	model.nNumofSV = nNumofSVs;
	model.pnIndexofSV = new int[nNumofSVs];
	//sv_coef is a second level pointer, which is for multiclasses SVM
	model.sv_coef = new float_point*[3];
	model.sv_coef[0] = new float_point[nNumofSVs]; 	//this coef is for GPU computation convenience
	model.sv_coef[1] = new float_point[nNumofPSV];	//this coef is for saving model (positive support vectors)
	model.sv_coef[2] = new float_point[nNumofNSV];	//this coef is for saving model (negative support vectors)
	memcpy(model.sv_coef[0], pfYiAlphaTemp, sizeof(float_point) * nNumofSVs);
	memcpy(model.sv_coef[1], pfPositiveAlphaTemp, sizeof(float_point) * nNumofPSV);
	memcpy(model.sv_coef[2], pfNegativeAlphaTemp, sizeof(float_point) * nNumofNSV);
	memcpy(model.pnIndexofSV, pnIndexofSVTemp, sizeof(int) * nNumofSVs);

	//compute bias
	model.rho = new float_point[1];
	if(nNumofFreeAlpha > 0)
	{
		model.rho[0] = (fYGsumofFreeAlpha / nNumofFreeAlpha);
	}
	else
	{
		model.rho[0] = (-m_pSMOSolver->m_fUpValue + m_pSMOSolver->m_fLowValue) / 2;
	}
	cout << "rho = " << model.rho[0] << endl;

	delete[] pnLabel;
	delete[] pfAlpha;
	delete[] pfYiAlphaTemp;
	delete[] pfPositiveAlphaTemp;
	delete[] pfNegativeAlphaTemp;
	delete[] pnIndexofSVTemp;
	delete[] pfYiFValue;

//	cout << m_pSMOSolver->m_fUpValue << " v.s. " << m_pSMOSolver->m_fLowValue << endl;
	//#####
	m_pSMOSolver->SMOSolverEnd();
	m_pSMOSolver->CleanCache();

	//can't find a optimal classifier
	if(nIter == nMaxIter)
	{
		bReturn = false;
	}
	return bReturn;
}

/*
 * @brief: set data involved in training, as in n-fold-cross validation, training data may be consisted of two parts
 * @param: nStart1: the start position of the first continuous piece of data
 * @param: nStart2: the start position of the second continuous piece of data
 * @param: nEnd: the last element in the part (the last element is included in the part)
 */
bool CSVMTrainer::SetInvolveTrainingData(int nStart1, int nEnd1, int nStart2, int nEnd2)
{
	bool bReturn = true;

	bReturn = m_pSMOSolver->SetInvolveData(nStart1, nEnd1, nStart2, nEnd2);

	return bReturn;
}

/*
 * @brief: save SVM model to file
 */
bool CSVMTrainer::SaveModel(string strFileName, const svm_model *model, vector<vector<float_point> >& v_vTrainingExample)
{
	bool bReturn = false;
	ofstream writeOut;
	writeOut.open(strFileName.c_str(), ios::out);

	if(!writeOut.is_open())
	{
		return bReturn;
	}

	bReturn = true;
	//these two output may need to be improved
	writeOut << "svm_type c_svc" << endl;
	writeOut << "kernel_type rbf" << endl;

	//stable output
	writeOut << "gamma " << gfGamma << endl;
	writeOut << "nr_class " << 2 << endl;
	writeOut << "total_sv " << model->nNumofSV << endl;
	writeOut << "rho " << model->rho[0] << endl;

	//have potential risk
	writeOut << "label " << model->label[0] << " " << model->label[1] << endl;
	writeOut << "nr_sv " << model->nSV[0] << " " << model->nSV[1] << endl;

	//data of support vectors
	writeOut << "SV" << endl;



	const float_point * const *sv_coef = model->sv_coef;
	int *pnIndexofSV = model->pnIndexofSV;
	int nNumofSVs = model->nNumofSV;

	for(int i = 0; i < nNumofSVs; i++)
	{
		writeOut << sv_coef[0][i] << " ";

		for(int j = 0; j < model->nDimension; j++)
		{
			writeOut << j << ":" << v_vTrainingExample[pnIndexofSV[i]][j]<< " ";
		}
/*			const svm_node *p = SV[i];

			if(param.kernel_type == PRECOMPUTED)
				fprintf(fp,"0:%d ",(int)(p->value));
			else
				while(p->index != -1)
				{
					fprintf(fp,"%d:%.8g ",p->index,p->value);
					p++;
				}
*/		writeOut << endl;
	}
	writeOut.close();
	return bReturn;
}
