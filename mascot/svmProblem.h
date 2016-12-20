/*
 * @author: Created by shijiashuai on 2016/11/1.
 */

#ifndef MASCOT_SVM_SVMPROBLEM_H
#define MASCOT_SVM_SVMPROBLEM_H

#include <vector>
#include <cusparse.h>
#include"../svm-shared/gpu_global_utility.h"

using std::vector;

class SvmProblem {
public:
    vector<vector<svm_node> > v_vSamples;
    vector<int> v_nLabels;
    vector<int> count;
    vector<int> start;
    vector<int> perm;
    vector<int> label;
    int numOfFeatures;
    bool subProblem;

    int getNumOfFeatures() const;

    //for subProblem
    vector<int> originalIndex;
    vector<int> originalLabel;

    SvmProblem(const vector<vector<svm_node> > &v_vSamples, int numOfFeatures, const vector<int> &v_nLabels) :
            v_vSamples(v_vSamples), v_nLabels(v_nLabels), numOfFeatures(numOfFeatures), subProblem(false){
        this->groupClasses();
    }

    void groupClasses();

    SvmProblem getSubProblem(int i, int j) const;
    vector<vector<svm_node> > getOneClassSamples(int i) const;

    unsigned int getNumOfClasses() const;

    unsigned long long getNumOfSamples() const;
};

class CSRMatrix {
public:
    const vector<vector<svm_node> > &samples;
    vector<float_point> csrVal;
    vector<float_point> csrValSelfDot;
    vector<int> csrRowPtr;
    vector<int> csrColInd;
    int numOfFeatures;

    int getNumOfFeatures() const;

    CSRMatrix(const vector<vector<svm_node> >&samples, int numOfFeatures);
    int getNnz() const;

    const float_point *getCSRVal() const;

    const float_point *getCSRValSelfDot() const;

    const int *getCSRRowPtr() const;

    const int *getCSRColInd() const;

    int getNumOfSamples() const;
    static void CSRmm2Dense(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB,
                            int m, int n, int k,
                            const cusparseMatDescr_t descrA,
                            const int nnzA, const float *valA, const int *rowPtrA, const int *colIndA,
                            const cusparseMatDescr_t descrB,
                            const int nnzB, const float *valB, const int *rowPtrB, const int *colIndB,
                            float *C);
};

#endif //MASCOT_SVM_SVMPROBLEM_H
