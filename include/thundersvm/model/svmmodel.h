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
    int max2power(int n) const;

    virtual void smo_solver(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
                            SyncData<real> &f_val, real eps, real C, int ws_size) const;

    virtual void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                                    const SyncData<real> &alpha, real C, SyncData<int> &working_set) const;

    virtual real
    calculate_rho(const SyncData<real> &alpha, const SyncData<real> &f_val, const SyncData<int> &y, real C) const;

    virtual void record_model(const SyncData<real> &alpha, const SyncData<int> &y, const DataSet::node2d &instances,
                              const SvmParam param);

    SvmParam param;
    vector<real> coef;
    DataSet::node2d sv;
    vector<int> sv_index;
    real rho;
    const char *kernel_type_name[6] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};
    const char *svm_type_name[6] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "NULL"};  /* svm_type */
private:
    void init_f(const SyncData<real> &alpha, const KernelMatrix &k_mat, SyncData<real> &f_val) const;
};

#endif //THUNDERSVM_SVMMODEL_H
