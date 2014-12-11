/**
 * modelSelector.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef MODELSELECTOR_H_
#define MODELSELECTOR_H_

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "kernelCalculater/kernelCalculater.h"
#include "svmPredictor.h"
#include "svmTrainer.h"
#include "gpu_global_utility.h"

class CModelSelector
{
public:
	CSVMTrainer *m_pTrainer;
	CSVMPredictor *m_pPredictor;

public:
	//initialize instance
	CModelSelector()
	{
		m_pTrainer = NULL;
		m_pPredictor = NULL;
	}
	~CModelSelector(){};

	bool GridSearch(const grid&, vector<vector<float_point> >&, vector<int> &vnLabel);
	bool CrossValidation(const int&, vector<int> &vnLabel, int *&);
	void PrecomputeKernelMatrix(vector<vector<float_point> > &v_vDocVector, CHessianIOOps *hessianIOOps);

private:
	bool DestroySVMModel(svm_model&);
	bool OutputResult(vector<int> &, int *, int nSizeofSamples);
};


#endif /* MODELSELECTOR_H_ */
