//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMPARAM_H
#define THUNDERSVM_SVMPARAM_H

#include "thundersvm.h"

struct SvmParam{
    enum SVM_TYPE {
        SVC, SVR, ONE_CLASS
    };
    real C;
    real gamma;
    real p;
    real nu;
    real epsilon;
    int task_type;
	char *argvi;
	char *argvi1;
	char *argvi2;

    int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	real coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};
enum {svmTrain, svmScale, svmPredict};	/*task_type*/
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
//struct SvmParam param_cmd; //for cmd parser
#endif //THUNDERSVM_SVMPARAM_H
