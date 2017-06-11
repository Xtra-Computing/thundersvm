//
// Created by ss on 16-11-2.
//

#ifndef MASCOT_SVM_SVMMODEL_H
#define MASCOT_SVM_SVMMODEL_H

#include <vector>
#include <cstdio>
#include <sstream>
#include <driver_types.h>
#include <helper_cuda.h>
#include <cuda.h>

#include"pthread.h"
#include "svmProblem.h"
#include "../svm-shared/csrMatrix.h"
#include "../SharedUtility/KeyValue.h"
using std::vector;
using std::string;

class SvmModel {
public:
    vector<int> label;					//class labels; its size equals to the number of classes.
    uint nrClass;
    int numOfFeatures;
    vector<vector<int> > svIndex;   //indices of SVs in each training subset
    vector<vector<KeyValue> > svMap;
    CSRMatrix *svMapCSRMat = NULL;
    vector<vector<real> > coef;
    vector<vector<real> > allcoef;
    vector<real> probA;
    vector<real> probB;
    real *devSVMapVal;
    real *devSVMapValSelfDot;
    int *devSVMapRowPtr;
    int *devSVMapColInd;
    real *devCoef = NULL;
    int *devStart = NULL;
    int *devCount = NULL;
    real *devRho = NULL;
    real *devProbA = NULL;
    real *devProbB = NULL;
    SVMParam param;
    uint cnr2;							//total number of svm models to train
    int *devSVIndex;
    bool probability;

    vector<vector<int> > missLabellingMatrix;	//for measuring classification error for each sub-classifier
    vector<real> vC;							//keep improving C
    vector<int> nSV;
    vector<bool> nonzero;//chen add
private:
    int numOfSVs;
    vector<int> start;					//for multiclass, start position for each class of instances
    vector<int> count;					//support vectors of the i-th class
    vector<real> rho;

    //device pointers
//    svm_node **devSVs = NULL;
    uint inline getK(int i, int j) const;

	//have changed the type of *dec_values,& A,& B
//	void gpu_sigmoid_train(int l, const float_point *dec_values, const float_point *labels,
//	float_point& A, float_point& B);

    void
    sigmoidTrain(const real *decValues, const int, const vector<int> &labels, real &A, real &B);

    void transferToDevice();

    //SvmModel has device pointer, so duplicating SvmModel is not allowed
    SvmModel &operator=(const SvmModel &);

    SvmModel(const SvmModel &);

public:
    ~SvmModel();
    SvmModel() {};

    void fit(const SvmProblem &problem, const SVMParam &param);
    void addBinaryModel(const SvmProblem &subProblem, const vector<int> &svLocalIndex,
    					const vector<real> &coef, real rho, int i, int j);
	void getModelParam(const SvmProblem &subProblem, const vector<int> &svIndex,const vector<real> &coef, 
	                            vector<int> &prob_start, int ci,int i, int j);
	void updateAllCoef(int l, int indOffset, int nr_class, int &count, int k, const vector<int> & svIndex, const vector<real> &coef,vector<int> &prob_start);
    bool isProbability() const;
	bool saveLibModel(string filename, const SvmProblem &problem);
	void loadLibModel(string filename, SvmModel &loadModel);
};

#endif //MASCOT_SVM_SVMMODEL_H
