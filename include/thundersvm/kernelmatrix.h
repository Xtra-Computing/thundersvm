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
    void get_rows(const SyncData<real> &dense_mat, const SyncData<real> &dot, SyncData<real> &kernel_rows,
                      int n_rows) const;
    void dns_csr_mul(const SyncData<real> &dense_mat, int n_rows, SyncData<real> &result) const;
    const SyncData<real> *diag() const;
    size_t m() const {return m_;};
    size_t n() const {return n_;};
    size_t nnz() const {return nnz_;};
private:
    SyncData<real> *val_;
    SyncData<int> *col_ind_;
    SyncData<int> *row_ptr_;
    SyncData<real> *diag_;
    SyncData<real> *self_dot_;
    size_t nnz_;
    size_t m_;
    size_t n_;
    real gamma;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
};
#endif //THUNDERSVM_KERNELMATRIX_H
