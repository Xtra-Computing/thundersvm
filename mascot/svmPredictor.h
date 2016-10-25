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
	bool ComputeYiAlphaKValue(float_point **pyfSVHessian, float_point *pfAlphaofSVs,
							  float_point *pfYiofSVs, float_point **pyfSVYiAlhpaHessian);

	bool SetInvolvePredictionData(int nStart1, int nEnd1);
	float_point* Predict(svm_model*, int *pnTestSampleId, const int&);
	float_point* Predict(svm_model*, svm_node **pInstance, const int &numInstance);
	float_point* ComputeClassLabel(int nNumofTestingSamples,
						   float_point *pfSVYiAlhpaHessian, const int &nNumofSVs,
						   float_point fBias, float_point *pfFinalResult);

	void ReadKVbasedOnSV(float_point *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs, int nNumofTestSamples);
	void ReadKVbasedOnTest(float_point *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs, int nNumofTestSamples);
private:
	void ReadFromHessian(float_point *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs,
						 int *pnTestSampleId, int nNumofTestSamples);
	void ComputeOnTheFly();
};

#endif /* SVMPREDICTOR_H_ */
