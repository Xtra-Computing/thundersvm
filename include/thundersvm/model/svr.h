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

    SVR(DataSet &dataSet, const SvmParam &svmParam);

    void train() override;


    void save_to_file(string path) override;

    void load_from_file(string path) override;

private:
    void record_model(const SyncData<real> &alpha, const SyncData<int> &y, real rho);
};

#endif //THUNDERSVM_SVR_H
