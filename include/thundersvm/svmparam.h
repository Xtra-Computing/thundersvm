//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMPARAM_H
#define THUNDERSVM_SVMPARAM_H

#include "thundersvm.h"

struct SvmParam {
    SvmParam() {
        svm_type = C_SVC;
        kernel_type = RBF;
        task_type = SVM_TRAIN;
        C = 1;
        gamma = 0;
        p = 0.1;
        epsilon = 0.001;
        nu = 0.5;
        probability = false;
        nr_weight = 0;
    }

    enum SVM_TYPE {
        C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
    };    /* svm_type */
    enum KERNEL_TYPE {
        LINEAR, POLY, RBF, SIGMOID/*, PRECOMPUTED*/
    }; /* kernel_type */
    enum TASK_TYPE {
        SVM_TRAIN, SVM_PREDICT
    };    /*task_type*/
    SVM_TYPE svm_type;
    KERNEL_TYPE kernel_type;
    TASK_TYPE task_type;

    real C; //for regularization
    real gamma; //for rbf kernel
    real p; //for regression
    real nu; //for nu-SVM
    real epsilon; //stopping criteria
    int degree; //degree for polynomial kernel

    real coef0;    /* for poly/sigmoid */

    /* these are for training only */
//    double cache_size; /* in MB */
    int nr_weight;        /* for C_SVC */
    int *weight_label;    /* for C_SVC */
    real *weight;        /* for C_SVC */
//    int shrinking;    /* use the shrinking heuristics */
    bool probability; /* do probability estimates */
    static const char *kernel_type_name[6];
    static const char *svm_type_name[6];  /* svm_type */
};
//struct SvmParam param_cmd; //for cmd parser
#endif //THUNDERSVM_SVMPARAM_H
