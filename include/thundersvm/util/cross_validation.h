//
// Created by jiashuai on 17-10-13.
//

#ifndef THUNDERSVM_CROSS_VALIDATION_H
#define THUNDERSVM_CROSS_VALIDATION_H

#include <thundersvm/thundersvm.h>
#include <thundersvm/model/svmmodel.h>

real cross_validation(SvmModel &model, const DataSet &dataset, SvmParam param, int n_fold);

#endif //THUNDERSVM_CROSS_VALIDATION_H
