/**
 * svmTrainer.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef SVMTRAINER_H_
#define SVMTRAINER_H_

#include <fstream>

#include "gpu_global_utility.h"
#include "kernelCalculater/kernelCalculater.h"
#include "constant.h"
#include "smoSolver.h"

using std::ofstream;

extern long nTimeOfUpdateAlpha;
extern long nTimeOfSelect1stSample;
extern long nTimeOfSelect2ndSample;
extern long nTimeOfUpdateYiFValue;
extern long nTimeOfLoop;
extern long nTimeOfPrep;
extern double dAverageLenS1;
extern double dAverageLenS2;
extern long nNumofCallSearch;

extern long nTimeofGetHessian;

class CSVMTrainer
{
public:
	//object member
	CSMOSolver *m_pSMOSolver;

public:
	//initialize instance
	//CSVMTrainer(){}
	CSVMTrainer(CSMOSolver *pSMOSolver)
	{
		m_pSMOSolver = pSMOSolver;
	}
	~CSVMTrainer(){}

	//train SVM model
	bool TrainModel(svm_model&, float_point *pfDevYiFValueSubset,
					float_point *pfDevAlphaSubset, int *pnDevLabelSubset, int, float_point*);

	void TrainStarting(int nNumofInstance, int nNumofTrainingExample,
					   float_point *pfDevYiFValueSubset, float_point *pfDevAlphaSubset, int *pnDevLabelSubset);
	void TrainEnding(int nIter, int nNumofTrainingExample, int nNumofInstance, svm_model &model,
			  	  	 int *pnDevLabelSubset, float_point *pfDevAlphaSubset, float_point *pfDevYiFValueSubset,
			  	  	 float_point *pfP);

	//set size of data participate in training
	bool SetInvolveTrainingData(int nStart1, int nEnd1, int nStart2, int nEnd2);
	/**** haven't implemented functions **********/

	//for future prediction purpose
	bool SaveModel(string strFileName, svm_model *model, vector<vector<float_point> >&);

	void setStream(const cudaStream_t &stream);

private:
	cudaStream_t stream;
};

#endif /* SVMTRAINER_H_ */
