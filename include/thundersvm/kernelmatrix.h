//
// Created by jiashuai on 17-9-19.
//

#ifndef THUNDERSVM_KERNELMATRIX_H
#define THUNDERSVM_KERNELMATRIX_H

#include <cusparse.h>
#include "thundersvm.h"
#include "syncdata.h"
#include "dataset.h"

class KernelMatrix{
public:
    explicit KernelMatrix(const DataSet::node2d &instances, size_t n_features, real gamma);
    ~KernelMatrix();
    void get_rows(const SyncData<int> &idx, SyncData<real> &kernel_rows) const;

    void get_rows(const DataSet::node2d &instances, SyncData<real> &kernel_rows) const;
    void dns_csr_mul(const SyncData<real> &dense_mat, int n_rows, SyncData<real> &result) const;
    const SyncData<real> *diag() const;

    size_t n_instances() const { return n_instances_; };//number of instances
    size_t n_features() const { return n_features_; };//number of features
    size_t nnz() const {return nnz_;};//number of nonzero
private:
    SyncData<real> *val_;
    SyncData<int> *col_ind_;
    SyncData<int> *row_ptr_;
    SyncData<real> *diag_;
    SyncData<real> *self_dot_;
    size_t nnz_;
    size_t n_instances_;
    size_t n_features_;
    real gamma;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
};
#endif //THUNDERSVM_KERNELMATRIX_H
