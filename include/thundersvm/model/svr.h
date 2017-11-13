//
// Created by jiashuai on 17-10-5.
//

#ifndef THUNDERSVM_SVR_H
#define THUNDERSVM_SVR_H

#include "thundersvm/thundersvm.h"
#include "svmmodel.h"
#include <map>

using std::map;

class SVR : public SvmModel {
public:
    void train(const DataSet &dataset, SvmParam param) override;

protected:
    void save_svr_coef(const SyncData<float_type> &alpha_2, const DataSet::node2d &instances);
};

#endif //THUNDERSVM_SVR_H
