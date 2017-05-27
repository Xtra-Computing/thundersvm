/**
 * svmPredictor.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef SVMPREDICTOR_H_
#define SVMPREDICTOR_H_

#include "../svm-shared/kernelCalculater/kernelCalculater.h"
#include "../svm-shared/gpu_global_utility.h"
#include "../svm-shared/HessianIO/baseHessian.h"
#include "classificationKernel.h"

using std::cerr;
/*
 * @brief: predictor class of SVM
 */
class CSVMPredictor
{
public:
	CKernelCalculater *m_pKernelCalculater;
	BaseHessian *m_pHessianReader;
	int m_nTestStart;
public:
	//CSVMPredictor(){}
	CSVMPredictor(BaseHessian *pHessianOps){m_pHessianReader = pHessianOps;}
	~CSVMPredictor(){}

	void SetCalculater(CKernelCalculater *p){m_pKernelCalculater = p;}
	bool ComputeYiAlphaKValue(real **pyfSVHessian, real *pfAlphaofSVs,
							  real *pfYiofSVs, real **pyfSVYiAlhpaHessian);

	bool SetInvolvePredictionData(int nStart1, int nEnd1);
	real* Predict(svm_model*, int *pnTestSampleId, const int&);
	real* Predict(svm_model*, svm_node **pInstance, int numInstance);
	real* ComputeClassLabel(int nNumofTestingSamples,
						   real *pfSVYiAlhpaHessian, const int &nNumofSVs,
						   real fBias, real *pfFinalResult);

	void ReadKVbasedOnSV(real *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs, int nNumofTestSamples);
	void ReadKVbasedOnTest(real *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs, int nNumofTestSamples);
private:
	int GetNumSV(svm_model *pModel);
	real* AllocateKVMem(int nNumofSVs, const int &nNumofTestSamples);
	real* PredictLabel(svm_model *pModel, int nNumofTestSamples, real *pfSVsKernelValues);

	void ReadFromHessian(real *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs,
						 int *pnTestSampleId, int nNumofTestSamples);
	void ComputeOnTheFly(real *pfSVsKernelValues, svm_model *pModel, svm_node **pInstance, int numInstance);
};

#endif /* SVMPREDICTOR_H_ */
