//
// Created by jiashuai on 17-10-6.
//

#ifndef THUNDERSVM_ONECLASS_SVC_H
#define THUNDERSVM_ONECLASS_SVC_H

#include "svmmodel.h"

/**
 * @brief Support Vector Machine for outlier detection (density estimation)
 */
class OneClassSVC : public SvmModel {
public:
    void train(const DataSet &dataset, SvmParam param) override;

    vector<float_type> predict(const DataSet::node2d &instances, int batch_size) override;

    ~OneClassSVC() override = default;

protected:
    void model_setup(const DataSet &dataset, SvmParam &param) override;

};

#endif //THUNDERSVM_ONECLASS_SVC_H
