//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMMODEL_H
#define THUNDERSVM_SVMMODEL_H

#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <thundersvm/kernelmatrix.h>
#include <map>

using std::map;

class SvmModel {
public:
    virtual void train(DataSet dataset, SvmParam param) {};

    virtual vector<real> predict(const DataSet::node2d &instances, int batch_size);

    virtual vector<real> cross_validation(DataSet dataset, SvmParam param, int n_fold);

    virtual void save_to_file(string path) = 0;

    virtual void load_from_file(string path) = 0;

protected:

    virtual void record_model(const SyncData<real> &alpha, const SyncData<int> &y, const DataSet::node2d &instances,
                              const SvmParam param);

    SvmParam param;
    vector<real> coef;
    DataSet::node2d sv;
    vector<int> sv_index;
    real rho;
    vector<real> c_weight;
    const char *kernel_type_name[6] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};
    const char *svm_type_name[6] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "NULL"};  /* svm_type */
};

#endif //THUNDERSVM_SVMMODEL_H
