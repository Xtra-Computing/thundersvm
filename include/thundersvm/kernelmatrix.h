//
// Created by jiashuai on 17-9-19.
//

#ifndef THUNDERSVM_KERNELMATRIX_H
#define THUNDERSVM_KERNELMATRIX_H

#include "thundersvm.h"
#include "syncdata.h"
#include "dataset.h"
#include "svmparam.h"

class KernelMatrix{
public:
    explicit KernelMatrix(const DataSet::node2d &instances, SvmParam param);

    void get_rows(const SyncData<int> &idx, SyncData<float_type> &kernel_rows) const;

    void get_rows(const DataSet::node2d &instances, SyncData<float_type> &kernel_rows) const;

    const SyncData<float_type> &diag() const;

    size_t n_instances() const { return n_instances_; };//number of instances
    size_t n_features() const { return n_features_; };//number of features
    size_t nnz() const {return nnz_;};//number of nonzero
private:
    KernelMatrix &operator=(const KernelMatrix &) const;

    KernelMatrix(const KernelMatrix &);

    SyncData<float_type> val_;
    SyncData<int> col_ind_;
    SyncData<int> row_ptr_;
    SyncData<float_type> diag_;
    SyncData<float_type> self_dot_;
    size_t nnz_;
    size_t n_instances_;
    size_t n_features_;
    SvmParam param;

    void dns_csr_mul(const SyncData<float_type> &dense_mat, int n_rows, SyncData<float_type> &result) const;

    void get_dot_product(const SyncData<int> &idx, SyncData<float_type> &dot_product) const;

    void get_dot_product(const DataSet::node2d &instances, SyncData<float_type> &dot_product) const;
};
#endif //THUNDERSVM_KERNELMATRIX_H
