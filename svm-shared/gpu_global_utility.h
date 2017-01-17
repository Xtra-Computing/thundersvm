/**
 * Utility.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef UTILITY_H_
#define UTILITY_H_

#include <vector>
#include "host_constant.h"
#include "svmParam.h"
#include "../mascot/DataIOOps/BaseLibsvmReader.h"

using std::vector;

extern float_point gfPCost;    //cost for positive samples in training SVM model (i.e., error tolerance)
extern float_point gfNCost;    //cost for negative samples in training SVM model
extern float_point gfGamma;

extern int gNTest;

/* a set of parameters which looks like a grid*/
struct Grid {
    vector<float_point> vfGamma;
    vector<float_point> vfC;
};

/* */
struct data_info {
    float_point *pfSampleData;        //a pointer to sample data (or index for pre-computed Hessian)
    float *pnLabel;                    //labels for each training sample
    int nNumofExample;                //the number of training samples
    int nNumofDim;                    //the number of dimensions
};

enum {
    LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
}; /* kernel_type */

//for sparse data
struct svm_node {
    int index;
    float_point value;
    svm_node(){};
    svm_node(int index, float_point value):
            index(index),value(value){};
};

class svm_model {
public:
    struct SVMParam param;    /* parameter */
    int nr_class;        /* number of classes, = 2 in regression/one class svm */
    int l;            /* total #SV */
    struct svm_node **SV;        /* SVs (SV[l]) */
    float_point **sv_coef;    /* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
    float_point *rho;        /* constants in decision functions (rho[k*(k-1)/2]) */
    float_point *probA;        /* pariwise probability information */
    float_point *probB;

    /* for classification only */

    int *label;        /* label of each class (label[k]) */
    int *nSV;        /* number of SVs for each class (nSV[k]) */
    /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    /* XXX */
    int free_sv;        /* 1 if svm_model is created by svm_load_model*/
    /* 0 if svm_model is created by svm_train */

    //new content
    int *pnIndexofSV;    //index of support vectors
    int nDimension;
};

#ifndef max

template<class T>
static inline T max(T x, T y) { return (x > y) ? x : y; }
template<class T>
static inline T min(T x, T y) { return (x < y) ? x : y; }

#endif

#define Ceil(a, b) (a%b!=0)?((a/b)+1):(a/b)

//#define CUDA_KERNEL_DIM(a, b)  <<< a, b >>>

#endif /* UTILITY_H_ */
