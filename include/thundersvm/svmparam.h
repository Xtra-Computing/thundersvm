//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMPARAM_H
#define THUNDERSVM_SVMPARAM_H

#include "thundersvm.h"

/**
 * @brief params for ThunderSVM
 */
struct SvmParam {
    SvmParam() {
        svm_type = C_SVC;
        kernel_type = RBF;
        C = 1;
        gamma = 0;
        p = 0.1f;
        epsilon = 0.001f;
        nu = 0.5;
        probability = false;
        nr_weight = 0;
        degree = 3;
        coef0 = 0;
        max_mem_size = static_cast<size_t>(8192) << 20;
    }

    /// SVM type
    enum SVM_TYPE {
        C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
    };
    /// kernel function type
    enum KERNEL_TYPE {
        LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
    };
    SVM_TYPE svm_type;
    KERNEL_TYPE kernel_type;

    ///regularization parameter
    float_type C;
    ///for RBF kernel
    float_type gamma;
    ///for regression
    float_type p;
    ///for \f$\nu\f$-SVM
    float_type nu;
    ///stopping criteria
    float_type epsilon;
    ///degree for polynomial kernel
    int degree;
    ///for polynomial/sigmoid kernel
    float_type coef0;
    ///for SVC
    int nr_weight;
    ///for SVC
    int *weight_label;
    ///for SVC
    float_type *weight;
    ///do probability estimates
    int probability;
    ///maximum memory size
    size_t max_mem_size;
    static const char *kernel_type_name[6];
    static const char *svm_type_name[6];
};
#endif //THUNDERSVM_SVMPARAM_H
