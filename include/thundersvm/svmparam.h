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
    SVM_TYPE svm_type;
    KERNEL_TYPE kernel_type;

    float_type C; //for regularization
    float_type gamma; //for rbf kernel
    float_type p; //for regression
    float_type nu; //for nu-SVM
    float_type epsilon; //stopping criteria
    int degree; //degree for polynomial kernel

    float_type coef0;    /* for poly/sigmoid */

    /* these are for training only */
//    double cache_size; /* in MB */
    int nr_weight;        /* for C_SVC */
    int *weight_label;    /* for C_SVC */
    float_type *weight;        /* for C_SVC */
//    int shrinking;    /* use the shrinking heuristics */
    int probability; /* do probability estimates */
    static const char *kernel_type_name[6];
    static const char *svm_type_name[6];  /* svm_type */
};
#endif //THUNDERSVM_SVMPARAM_H
