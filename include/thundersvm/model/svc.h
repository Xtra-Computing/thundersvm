//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVC_H
#define THUNDERSVM_SVC_H

#include <map>
#include <thundersvm/kernelmatrix.h>
#include "svmmodel.h"

using std::map;

class SVC : public SvmModel {
public:

    SVC() : SvmModel() {
        param.svm_type = SvmParam::SVC;
    }

    void train(DataSet dataset, SvmParam param) override;

    vector<real> predict(const DataSet::node2d &instances, int batch_size) override;

    void save_to_file(string path) override;

    void load_from_file(string path) override;

private:

    void record_binary_model(int k, const SyncData<real> &alpha, const SyncData<int> &y, real rho,
                             const vector<int> &original_index, const DataSet::node2d &original_instance);

    map<int, int> sv_index_map;
    vector<vector<real>> coef; //alpha_i * y_i
    DataSet::node2d sv;
    vector<vector<int>> sv_index;
    vector<real> rho;
    vector<int> label;

    size_t n_classes;
    size_t n_binary_models;
};

#endif //THUNDERSVM_SVC_H
