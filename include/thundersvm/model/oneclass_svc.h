//
// Created by jiashuai on 17-10-6.
//

#ifndef THUNDERSVM_ONECLASS_SVC_H
#define THUNDERSVM_ONECLASS_SVC_H

#include "svmmodel.h"

class OneClassSVC : public SvmModel {
public:
    void train(const DataSet &dataset, SvmParam param) override;

    vector<real> predict(const DataSet::node2d &instances, int batch_size) override;

};

#endif //THUNDERSVM_ONECLASS_SVC_H
