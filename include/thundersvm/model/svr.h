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

    vector<real> predict(const DataSet::node2d &instances, int batch_size) override;

    void save_to_file(string path) override;

    void load_from_file(string path) override;

private:
    void record_model(const SyncData<real> &alpha, const SyncData<int> &y, real rho);

    vector<real> coef;
    vector<real> alpha;
    DataSet::node2d sv;
    vector<int> sv_index;
    map<int, int> sv_map;
    real rho;
    int n_instances;
};

#endif //THUNDERSVM_SVR_H
