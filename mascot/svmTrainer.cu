/*
 * @brief: this file contains the definition of svm trainer class
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 */

#include "../svm-shared/svmTrainer.h"
#include "time.h"
#include "../svm-shared/gpu_global_utility.h"
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
bool CSVMTrainer::TrainModel(svm_model &model, real *pfDevYiFValueSubset, real *pfDevAlphaSubset,
							 int *pnDevLabelSubset, int nNumofInstance, real *pfP){
	bool bReturn = true;

	assert(nNumofInstance > 0  && pfDevAlphaSubset != NULL &&
		   pfDevYiFValueSubset != NULL && pnDevLabelSubset != NULL);

	/*************** prepare to perform training *************/
	TrainStarting(nNumofInstance, nNumofInstance, pfDevYiFValueSubset, pfDevAlphaSubset, pnDevLabelSubset);

	//start training process
	int nIter = 0;
	int nMaxIter = (nNumofInstance > INT_MAX / ITERATION_FACTOR ? INT_MAX : ITERATION_FACTOR * nNumofInstance) * 4;
	int nSelectFirstSample = -1;
	int nSelectSecondeSample = -1;

	timespec timeLoopS, timeLoopE;
	clock_gettime(CLOCK_REALTIME, &timeLoopS);

	while(nIter < nMaxIter)
	{
		int nEnd = m_pSMOSolver->Iterate(pfDevYiFValueSubset, pfDevAlphaSubset, pnDevLabelSubset, nNumofInstance);

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
	clock_gettime(CLOCK_REALTIME, &timeLoopE);
	long lTempLoop = ((timeLoopE.tv_sec - timeLoopS.tv_sec) * 1e9 + (timeLoopE.tv_nsec - timeLoopS.tv_nsec));
	if(lTempLoop > 0)
		nTimeOfLoop += lTempLoop;
	else
		cout << "loop timer error" << endl;

	TrainEnding(nIter, nNumofInstance, nNumofInstance, model,
				pnDevLabelSubset, pfDevAlphaSubset, pfDevYiFValueSubset, pfP);

	//can't find a optimal classifier
	if(nIter == nMaxIter)
	{
		bReturn = false;
	}
	return bReturn;
}
