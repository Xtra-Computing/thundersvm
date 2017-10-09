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
    SvmModel(DataSet &dataSet, const SvmParam &svmParam);

    virtual void train() = 0;

    virtual vector<real> predict(const DataSet::node2d &instances, int batch_size);

    virtual void save_to_file(string path) = 0;

    virtual void load_from_file(string path) = 0;

protected:
    int max2power(int n) const;

    virtual void smo_solver(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
                            SyncData<real> &f, real eps, real C, int ws_size) const;

    virtual void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                                    const SyncData<real> &alpha, SyncData<int> &working_set) const;

    virtual void record_model(const SyncData<real> &alpha, const SyncData<int> &y);

    DataSet &dataSet;
    SvmParam svmParam;
    vector<real> coef;
    DataSet::node2d sv;
    vector<int> sv_index;
    real rho;
    int n_instances;
};

#endif //THUNDERSVM_SVMMODEL_H
