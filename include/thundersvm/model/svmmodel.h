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
    virtual void train(DataSet dataset, SvmParam param) = 0;

    virtual vector<real> predict(const DataSet::node2d &instances, int batch_size);

    virtual vector<real> cross_validation(DataSet dataset, SvmParam param, int n_fold);

    virtual void save_to_file(string path);

    virtual void load_from_file(string path);

protected:

    virtual void record_model(const SyncData<real> &alpha, const SyncData<int> &y, const DataSet::node2d &instances,
                              const SvmParam param);

    SvmParam param;
    vector<real> coef;
    DataSet::node2d sv;
    vector<int> sv_index;
    real rho;
};

#endif //THUNDERSVM_SVMMODEL_H
