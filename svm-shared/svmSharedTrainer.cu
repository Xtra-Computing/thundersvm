/*
 * @brief: this file contains the definition of svm trainer class
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 */

#include "svmTrainer.h"
#include "storageManager.h"
#include <sys/time.h>
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

/**
 * @brief: training preparation, e.g. allocate some resources
 */
void CSVMTrainer::TrainStarting(int nNumofInstance, int nNumofTrainingExample,
								real *pfDevYiFValueSubset, real *pfDevAlphaSubset, int *pnDevLabelSubset)
{
	timespec timePrepS, timePrepE;
	clock_gettime(CLOCK_REALTIME, &timePrepS);

	//allocate memory for SMO slover
	bool bPreperation = m_pSMOSolver->SMOSolverPreparation(nNumofTrainingExample);
	assert(bPreperation != false);

	StorageManager *manager = StorageManager::getManager();
	int nSizeofCache = manager->RowInGPUCache(nNumofTrainingExample, nNumofInstance);

	timeval tInit1, tInit2;
	real InitCacheElapsedTime;
	gettimeofday(&tInit1, NULL);
	//initialize cache
	cout << "cache size v.s. ins is " << nSizeofCache << " v.s. " << nNumofInstance << endl;
	bool bInitCache = m_pSMOSolver->InitCache(nSizeofCache, nNumofInstance);
	assert(bInitCache != false);
	cout << "cache initialized" << endl;
	gettimeofday(&tInit2, NULL);
	InitCacheElapsedTime = (tInit2.tv_sec - tInit1.tv_sec) * 1000.0;
	InitCacheElapsedTime += (tInit2.tv_usec - tInit1.tv_usec) / 1000.0;
	cout << "initCache time: " << InitCacheElapsedTime << " ms.\n";

	clock_gettime(CLOCK_REALTIME, &timePrepE);
	long lTempPrep = ((timePrepE.tv_sec - timePrepS.tv_sec) * 1e9 + (timePrepE.tv_nsec - timePrepS.tv_nsec));
	if(lTempPrep > 0)
		nTimeOfPrep += lTempPrep;
	else
		cout << "preparation timer error" << endl;

	//allocate memory for reading hessian row
	m_pSMOSolver->m_pHessianReader->AllocateBuffer(1);

	checkCudaErrors(cudaMemcpy(m_pSMOSolver->m_pnLabel, pnDevLabelSubset,
					sizeof(int) * nNumofTrainingExample, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_pSMOSolver->m_pfGValue, pfDevYiFValueSubset,
					sizeof(real) * nNumofTrainingExample, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemset(pfDevAlphaSubset, 0,	sizeof(real) * nNumofTrainingExample));
}

/*
 * @brief: clean resources when the training is terminated
 * @pfP: is used in computing the objective value
 */
void CSVMTrainer::TrainEnding(int nIter, int nNumofTrainingExample, int nNumofInstance, svm_model &model,
							  int *pnDevLabelSubset, real *pfDevAlphaSubset, real *pfDevYiFValueSubset,
							  real *pfP)
{
//	cudaStreamDestroy(m_pSMOSolver->m_stream1_Hessian_row);//destroy stream
	//release buffer for reading hessian row
	m_pSMOSolver->m_pHessianReader->ReleaseBuffer();

	//m_pSMOSolver->m_pCache->PrintCache();

	cout << "# of iteration: " << nIter << endl;

	//store classification result in SVM Model
	int *pnLabel = new int[nNumofTrainingExample];
	real *pfAlpha = new real[nNumofTrainingExample];
	real *pfYiFValue = new real[nNumofTrainingExample];
	memset(pnLabel, 0, sizeof(int) * nNumofTrainingExample);
	memset(pfAlpha, 0, sizeof(real) * nNumofTrainingExample);
	memset(pfYiFValue, 0, sizeof(real) * nNumofTrainingExample);

	checkCudaErrors(cudaMemcpy(pnLabel, pnDevLabelSubset, sizeof(int) * nNumofTrainingExample, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pfAlpha, pfDevAlphaSubset, sizeof(real) * nNumofTrainingExample, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pfYiFValue, pfDevYiFValueSubset, sizeof(real) * nNumofTrainingExample, cudaMemcpyDeviceToHost));

	//compute the # of support vectors, and get bias of the model
	real *pfYiAlphaTemp = new real[nNumofTrainingExample];
	real *pfPositiveAlphaTemp = new real[nNumofTrainingExample];
	real *pfNegativeAlphaTemp = new real[nNumofTrainingExample];
	int *pnIndexofSVTemp = new int[nNumofTrainingExample];

	real fYGsumofFreeAlpha = 0;
	int nNumofFreeAlpha = 0;
	int nNumofSVs = 0;
	int nNumofPSV = 0, nNumofNSV = 0;

	// calculate objective value
	if(pfP != NULL)
	{
		float v = 0;
		for(int i=0;i<nNumofTrainingExample;i++)
			v += pfAlpha[i] * pnLabel[i]* (pfYiFValue[i] + pfP[i]);

		cout << "obj =" <<  v/2 << endl;
	}

	float sum_alpha = 0;
	for(int i = 0; i < nNumofInstance; i++)
	{
		//pfAlpha[i] = pfAlpha[i] - pfAlpha[i+nNumofInstance]; //this line is for regression
		sum_alpha += fabs(pfAlpha[i]);
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
		if(pfAlpha[i] != 0)
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
	}

	cout << "nu=" << sum_alpha/nNumofInstance << endl;
	cout << "# free SV " << nNumofFreeAlpha << "; # of SV " << nNumofSVs << endl;
	/**************** store result to SVM model ***********************/
	model.label = new int[2];
	model.nSV = new int[2];
	model.label[0] = 1;
	model.label[1] = -1;
	model.nSV[0] = nNumofPSV;
	model.nSV[1] = nNumofNSV;

	model.pnIndexofSV = new int[nNumofSVs];
	model.SV = new svm_node*[nNumofSVs];
	//sv_coef is a second level pointer, which is for multiclasses SVM
	model.sv_coef = new real*[3];
	model.sv_coef[0] = new real[nNumofSVs]; 	//this coef is for GPU computation convenience
	model.sv_coef[1] = new real[nNumofPSV];	//this coef is for saving model (positive support vectors)
	model.sv_coef[2] = new real[nNumofNSV];	//this coef is for saving model (negative support vectors)
	memcpy(model.sv_coef[0], pfYiAlphaTemp, sizeof(real) * nNumofSVs);
	memcpy(model.sv_coef[1], pfPositiveAlphaTemp, sizeof(real) * nNumofPSV);
	memcpy(model.sv_coef[2], pfNegativeAlphaTemp, sizeof(real) * nNumofNSV);
	memcpy(model.pnIndexofSV, pnIndexofSVTemp, sizeof(int) * nNumofSVs);

	//compute bias
	model.rho = new real[1];
	/*if(nNumofFreeAlpha > 0)
	{
		model.rho[0] = (fYGsumofFreeAlpha / nNumofFreeAlpha);
	}
	else
	{*/
		model.rho[0] = (-m_pSMOSolver->upValue + m_pSMOSolver->m_fLowValue) / 2;
	//}

	delete[] pnLabel;
	delete[] pfAlpha;
	delete[] pfYiAlphaTemp;
	delete[] pfPositiveAlphaTemp;
	delete[] pfNegativeAlphaTemp;
	delete[] pnIndexofSVTemp;
	delete[] pfYiFValue;

	cout << m_pSMOSolver->upValue << " v.s. " << m_pSMOSolver->m_fLowValue << endl;
	cout << "bias=" << model.rho[0] << endl;
	//#####
	m_pSMOSolver->SMOSolverEnd();
	m_pSMOSolver->CleanCache();
}

/*
 * @brief: save SVM model to file
 */
bool CSVMTrainer::SaveModel(string strFileName, svm_model *model, vector<vector<real> >& v_vTrainingExample)
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
	writeOut << "total_sv " << model->nSV[0] + model->nSV[1] << endl;
	writeOut << "rho " << model->rho[0] << endl;

	//have potential risk
	writeOut << "label " << model->label[0] << " " << model->label[1] << endl;
	writeOut << "nr_sv " << model->nSV[0] << " " << model->nSV[1] << endl;

	//data of support vectors
	writeOut << "SV" << endl;

	const real * const *sv_coef = model->sv_coef;
	int *pnIndexofSV = model->pnIndexofSV;
	int nNumofSVs = model->nSV[0] + model->nSV[1];//Better to use as a function (similar to GetNumSV as in svmPredictor.cu)
	model->l = nNumofSVs;

	for(int i = 0; i < nNumofSVs; i++)
	{
		writeOut << sv_coef[0][i] << " ";

		int nonzeroDimCounter = 0;
		for(int j = 0; j < v_vTrainingExample[0].size(); j++)//for each of dimension
		{
			if(v_vTrainingExample[pnIndexofSV[i]][j] > 0)//for the support vector i
			{
				writeOut << j << ":" << v_vTrainingExample[pnIndexofSV[i]][j]<< " ";
				nonzeroDimCounter++;
			}
		}
		model->SV[i] = new svm_node[nonzeroDimCounter + 1];//the last node to indicate the end of SV

		int curDimCounter = 0;
		for(int j = 0; j < v_vTrainingExample[0].size(); j++)//for each of dimension
		{
			if(v_vTrainingExample[pnIndexofSV[i]][j] > 0)//for the support vector i
			{
				model->SV[i][curDimCounter++].index = j;
				model->SV[i][curDimCounter++].value = v_vTrainingExample[pnIndexofSV[i]][j];
			}
		}
		//set the end of support vect
		model->SV[i][curDimCounter++].index = -1;
		model->SV[i][curDimCounter++].value = -1;

		writeOut << endl;
	}
	writeOut.close();
	return bReturn;
}

