
#include <helper_cuda.h>
#include "modelSelector.h"
#include "../svm-shared/svmTrainer.h"
#include "../svm-shared/HessianIO/baseHessian.h"
#include "../svm-shared/HessianIO/parAccessor.h"
#include "../svm-shared/HessianIO/seqAccessor.h"
#include "../svm-shared/storageManager.h"
#include "../SharedUtility/DataType.h"

/**
 * @brief: search the best pair of parameters
 */
bool CModelSelector::GridSearch(const Grid &SGrid, vector<vector<real> > &v_vDocVector, vector<int> &vnLabel)
{
	bool bReturn = false;

	const vector<real> &vfGamma = SGrid.vfGamma;
	const vector<real> &vfC = SGrid.vfC;

	int nNumofSample = v_vDocVector.size();
	int *pnPredictedLabel = new int[nNumofSample];
	int nNumofFold = 10;//10 means 10-fold cross

	DeviceHessian::m_nTotalNumofInstance = v_vDocVector.size();
	CLRUCache cacheStrategy(v_vDocVector.size());
	cout << "using " << cacheStrategy.GetStrategy() << " caching strategy"<< endl;

	ofstream confusion;
	confusion.open("matrix.txt", std::ofstream::out | std::ofstream::app);

	for(uint j = 0; j < vfGamma.size(); j++)
	{
		CRBFKernel rbf(vfGamma[j]);//ignore
		DeviceHessian hessianIOOps(&rbf);
		//initial Hessian accessor
		SeqAccessor accessor;
		accessor.m_nTotalNumofInstance = DeviceHessian::m_nTotalNumofInstance;
		accessor.SetInvolveData(0, accessor.m_nTotalNumofInstance - 1, -1, -1);

		hessianIOOps.SetAccessor(&accessor);

		CSVMPredictor svmPredicter(&hessianIOOps);

		hessianIOOps.PrecomputeKernelMatrix(v_vDocVector, &hessianIOOps);

		//start n-fold-cross-validation, by changing C for SVM
		for(uint k = 0; k < vfC.size(); k++)
		{
			CSMOSolver s(&hessianIOOps, &cacheStrategy);

			CSVMTrainer svmTrainer(&s);
			m_pTrainer = &svmTrainer;
			m_pPredictor = &svmPredicter;

			memset(pnPredictedLabel, 0, sizeof(int) * nNumofSample);
			gfNCost = vfC[k];
			gfPCost = vfC[k];
			gfGamma = vfGamma[j];
			ofstream writeOut(OUTPUT_FILE, ios::app | ios::out);
			writeOut << "Gamma=" << vfGamma[j] << "; Cost=" << vfC[k] << endl;

			timespec timeValidS, timeValidE;
			clock_gettime(CLOCK_REALTIME, &timeValidS);
			bool bCrossValidation = CrossValidation(nNumofFold, vnLabel, pnPredictedLabel);
			if(bCrossValidation == false)
			{
				cerr << "can't have valid result in N_fold_cross_validation" << endl;
				continue;
			}
			clock_gettime(CLOCK_REALTIME, &timeValidE);
			long lCrossValidationTime = ((timeValidE.tv_sec - timeValidS.tv_sec) * 1e9 + (timeValidE.tv_nsec - timeValidS.tv_nsec));
			writeOut.close();
			//output n-fold-cross-validation result
			OutputResult(confusion, vnLabel, pnPredictedLabel, nNumofSample);

			cout << "total time: " << (double)lCrossValidationTime / 1000000 << "ms" << endl;
		}//end varying C
		//release pinned memory
		cudaFreeHost(DeviceHessian::m_pfHessianRowsInHostMem);
		delete[] DeviceHessian::m_pfHessianDiag;
	}//end varying gamma

	delete[] pnPredictedLabel;

	return bReturn;
}

/*
 * @brief: n fold cross validation
 * @param: nFold: the number of fold for the cross validation
 */
bool CModelSelector::CrossValidation(const int &nFold, vector<int> &vnLabel, int *&pnPredictedLabel)
{
	bool bReturn = true;

	int nTotalNumofSamples = vnLabel.size();
	//get labels of data
	int *pnLabelAll = new int[nTotalNumofSamples];
	for(int l = 0; l < nTotalNumofSamples; l++)
	{
		if(vnLabel[l] != 1 && vnLabel[l] != -1)
		{
			cerr << "error label (valid label is -1 or 1): " << vnLabel[l] << endl;
			cerr << "multi-class problems are not supported" << endl;
			exit(0);
		}
		pnLabelAll[l] = vnLabel[l];
	}

	//check input parameters
	if((nFold < 1) || nTotalNumofSamples < nFold)
	{
		cerr << "error in cross validation: invalid parameters" << endl;
		exit(0);
	}

	//divide the training samples in to n folds. note that the last fold may be larger than other folds.
	int nSizeofFold = 0;
	nSizeofFold = nTotalNumofSamples / nFold;
	int *pnFoldStart = new int[nFold];
	//Initialise the first fold
	pnFoldStart[0] = 0;
	//for the case that there is only one fold
	if(nFold == 1)
	{
		pnFoldStart[1] = 0;
	}
	//start counting the size of each fold
	for(int i = 1; i < nFold; i++)
	{
		pnFoldStart[i] = pnFoldStart[i - 1] + nSizeofFold;
	}

	//during n-fold cross validation, training samples are divided into at most 2 parts
	int *pnSizeofParts = new int[2];

	/* allocate GPU device memory */
	//set default value at
	real *pfAlphaAll;
	real *pfYiGValueAll;
	pfAlphaAll = new real[nTotalNumofSamples];
	pfYiGValueAll = new real[nTotalNumofSamples];
	for(int i = 0; i < nTotalNumofSamples; i++)
	{
		//initially, the values of alphas are 0s
		pfAlphaAll[i] = 0;
		//GValue is -y_i, as all alphas are 0s. YiGValue is always -1
		pfYiGValueAll[i] = -pnLabelAll[i];
	}

	/* start n-fold-cross-validation */
	//allocate GPU memory for part of samples that are used to perform training.
	real *pfDevAlphaSubset;
	real *pfDevYiGValueSubset;
	int *pnDevLabelSubset;

	real *pfPredictionResult = new real[nTotalNumofSamples];
	for(int i = 0; i < nFold; i++)
	{
		/**************** training *******************/
		//first continual part of sample data
		if(i != 0)
		{
			pnSizeofParts[0] = pnFoldStart[i];
		}
		else
		{
			pnSizeofParts[0] = 0;
		}

		//second continual
		if(i != nFold - 1 || nFold == 1)//nFold == 1 is  for special case, where all samples are for training and testing
		{
			pnSizeofParts[1] = nTotalNumofSamples - pnFoldStart[i + 1];
		}
		else
		{
			pnSizeofParts[1] = 0;
		}

		//get size of training samples
		int nNumofTrainingSamples = 0;
		nNumofTrainingSamples = pnSizeofParts[0] + pnSizeofParts[1];

		//in n-fold-cross validation, the first (n -1) parts have the same size, so we can reuse memory
		if(i == 0 || (i == nFold - 1))
		{
			checkCudaErrors(cudaMalloc((void**)&pfDevAlphaSubset, sizeof(real) * nNumofTrainingSamples));
			checkCudaErrors(cudaMalloc((void**)&pfDevYiGValueSubset, sizeof(real) * nNumofTrainingSamples));
			checkCudaErrors(cudaMalloc((void**)&pnDevLabelSubset, sizeof(int) * nNumofTrainingSamples));
		}
		//set GPU memory
		checkCudaErrors(cudaMemset(pfDevAlphaSubset, 0, sizeof(real) * nNumofTrainingSamples));
		checkCudaErrors(cudaMemset(pfDevYiGValueSubset, -1, sizeof(real) * nNumofTrainingSamples));
		checkCudaErrors(cudaMemset(pnDevLabelSubset, 0, sizeof(int) * nNumofTrainingSamples));
		//copy training information to GPU for current training
		checkCudaErrors(cudaMemcpy(pfDevAlphaSubset, pfAlphaAll,
								   sizeof(real) * pnSizeofParts[0], cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pfDevYiGValueSubset, pfYiGValueAll,
								   sizeof(real) * pnSizeofParts[0], cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pnDevLabelSubset, pnLabelAll,
								   sizeof(int) * pnSizeofParts[0], cudaMemcpyHostToDevice));
		//part two
		if(pnSizeofParts[1] != 0)
		{
			checkCudaErrors(cudaMemcpy(pfDevAlphaSubset + pnSizeofParts[0], pfAlphaAll + pnFoldStart[i + 1],
									   sizeof(real) * pnSizeofParts[1], cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(pfDevYiGValueSubset + pnSizeofParts[0], pfYiGValueAll + pnFoldStart[i + 1],
									   sizeof(real) * pnSizeofParts[1], cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(pnDevLabelSubset + pnSizeofParts[0], pnLabelAll + pnFoldStart[i + 1],
									   sizeof(int) * pnSizeofParts[1], cudaMemcpyHostToDevice));
		}

		/************** train SVM model **************/
		int nSampleStart1, nSampleEnd1, nSampleStart2, nSampleEnd2;
		if(pnSizeofParts[0] != 0)
		{
			nSampleStart1 = 0;
			nSampleEnd1 = pnSizeofParts[0] - 1;
		}
		else
		{
			nSampleStart1 = -1;
			nSampleEnd1 = -1;
		}
		if(pnSizeofParts[1] != 0)
		{
			nSampleStart2 = pnFoldStart[i + 1];
			nSampleEnd2 = nTotalNumofSamples - 1;
		}
		else
		{
			nSampleStart2 = -1;
			nSampleEnd2 = -1;
		}

		cout << "training the " << i + 1 << "th classifier";
		cout.flush();

		svm_model model;
		//set data involved in training
		m_pTrainer->SetInvolveTrainingData(nSampleStart1, nSampleEnd1, nSampleStart2, nSampleEnd2);
		bool bTrain = m_pTrainer->TrainModel(model, pfDevYiGValueSubset, pfDevAlphaSubset, pnDevLabelSubset, nNumofTrainingSamples, NULL);
		if(bTrain == false)
		{
			cerr << "can't find an optimal classifier" << endl;
			bReturn = false;
			break;
		}

		/******************** prediction *******************/
		//get the size of a fold for testing
		int nNumofTestingSample = 0;
		if(i != nFold - 1)
			nNumofTestingSample = pnFoldStart[i + 1] - pnFoldStart[i];
		else
			nNumofTestingSample = nTotalNumofSamples - pnFoldStart[i];

		//get testing sample id
		int *pnTestSampleId;
		if(i == 0 || (i == nFold - 1))
		{
			pnTestSampleId = new int[nNumofTestingSample];
		}
		for(int j = 0; j < nNumofTestingSample; j++)
		{
			pnTestSampleId[j] = pnFoldStart[i] + j;
		}

		timespec timeClassificationS, timeClassificationE;
		clock_gettime(CLOCK_REALTIME, &timeClassificationS);

		cout << "performing classification...";
		//set data involve in prediction
		m_pPredictor->SetInvolvePredictionData(pnTestSampleId[0], pnTestSampleId[nNumofTestingSample - 1]);
		//perform prediction
		real *pfPartialPredictionResult;
		pfPartialPredictionResult = m_pPredictor->Predict(&model, pnTestSampleId, nNumofTestingSample);
		cout << " Done"<< endl;
		clock_gettime(CLOCK_REALTIME, &timeClassificationE);

		//copy partial result to global result
		for(int j = 0; j < nNumofTestingSample; j++)
		{
			pfPredictionResult[pnFoldStart[i] + j] = pfPartialPredictionResult[j];
		}

		//for comparing the results of the other gpu svm
		int nCorrect = 0;
		for(int j = 0; j < nNumofTestingSample; j++)
		{
			if(pfPartialPredictionResult[j] > 0 && pnLabelAll[pnFoldStart[i] + j] > 0)
				nCorrect++;
			else if(pfPartialPredictionResult[j] < 0 && pnLabelAll[pnFoldStart[i] + j] < 0)
				nCorrect++;
		}
		cout << "accuracy in this fold: " << (float)nCorrect/nNumofTestingSample << endl;

		delete[] pfPartialPredictionResult; //as memory is allocated during prediction
		//release memory, in the first (nFold - 2) iterations, the space of pnTestSampleId can be reused
		if(i >= (nFold - 2))
		{
			delete[] pnTestSampleId;
		}
		DestroySVMModel(model);
	}

	//calculate classification label
	for(int i = 0; i < nTotalNumofSamples; i++)
	{
		if(pfPredictionResult[i] > 0)
		{
			pnPredictedLabel[i] = 1;
		}
		else
		{
			pnPredictedLabel[i] = -1;
		}
	}

	checkCudaErrors(cudaFree(pfDevAlphaSubset));
	checkCudaErrors(cudaFree(pnDevLabelSubset));
	checkCudaErrors(cudaFree(pfDevYiGValueSubset));

	delete[] pfAlphaAll;
	delete[] pfYiGValueAll;

	delete[] pfPredictionResult;
	delete[] pnSizeofParts;

	delete[] pnFoldStart;
	return bReturn;
}

/*
 * @brief: output prediction result (e.g., accuracy, recall, precision etc.)
 * @param: pnOriginalLabel: label of training samples
 * @param: pnPredictedLabel: label assigned by SVM
 */
bool CModelSelector::OutputResult(ofstream &confusion, vector<int> &pnOriginalLabel, int *pnPredictedLabel, int nSizeofSample)
{
	bool bReturn = false;
	int nCorrect = 0, nTrueP = 0, nFalseP = 0, nFalseN = 0, nTrueN = 0;
	for(int i = 0; i < nSizeofSample; i++)
    {
		if((pnPredictedLabel[i] == 1 && pnOriginalLabel[i] == 1) || (pnPredictedLabel[i] == -1 && pnOriginalLabel[i] == -1))
		{
			nCorrect++;
		}

		if(pnPredictedLabel[i] == 1 && pnOriginalLabel[i] == 1)
		{
			nTrueP++;
		}
		else if(pnPredictedLabel[i] == -1 && pnOriginalLabel[i] == 1)
		{
			nFalseN++;
		}
		else if(pnPredictedLabel[i] == 1 && pnOriginalLabel[i] == -1)
		{
			nFalseP++;
		}
		else if(pnPredictedLabel[i] == -1 && pnOriginalLabel[i] == -1)
		{
			nTrueN++;
		}
		else
		{
			cout << "error in output result: " << pnPredictedLabel[i] << " and " << pnOriginalLabel[i]
			     << " are not equal to +1 or -1" << endl;
		}
    }

	double dPrecision = (double)nTrueP / (nTrueP + nFalseP);
	double dRecall = (double)nTrueP / (nFalseN + nTrueP);

	confusion << "TP: " << nTrueP << " FN: " << nFalseN << ";\t";
	confusion << "FP: " << nFalseP<< " TN: " << nTrueN << endl;

	cout << "Accuracy: " << (double)nCorrect / nSizeofSample << " precision@pos: " << dPrecision << " recall@pos: " << dRecall << endl;
	cout << "Precision@neg: " << (double)nTrueN/(nTrueN + nFalseN) << " recall@neg: " << (double)nTrueN/(nTrueN + nFalseP) << endl;

/*	ofstream writeOut(OUTPUT_FILE, ios::app | ios::out);
	writeOut << "true positive: " << nTrueP << "; true negative: " << nTrueN
			 << "; false positive: " << nFalseP << "; false negative: " << nFalseN << endl;
	writeOut << (double)nCorrect / nSizeofSample << " precision: " << dPrecision
				<< " recall: " << dRecall << endl;
	writeOut.close();
*/
	return bReturn;
}

/*
 * @brief: destroy svm model
 */
bool CModelSelector::DestroySVMModel(svm_model &model)
{
	bool bReturn = true;

	delete[] model.label;
	delete[] model.pnIndexofSV;
	delete[] model.rho;
	delete[] model.sv_coef[0];
	delete[] model.sv_coef[1];
	delete[] model.sv_coef[2];
	delete[] model.sv_coef;

	return bReturn;
}
