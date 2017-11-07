//
// Created by jiashuai on 17-9-20.
//

#ifndef THUNDERSVM_KERNELMATRIX_KERNEL_H
#define THUNDERSVM_KERNELMATRIX_KERNEL_H

#include "thundersvm/thundersvm.h"
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncdata.h>

namespace svm_kernel {
    void get_working_set_ins(const SyncData<real> &val, const SyncData<int> &col_ind, const SyncData<int> &row_ptr,
                             const SyncData<int> &data_row_idx, SyncData<real> &data_rows, int m);

    void
    RBF_kernel(const SyncData<real> &self_dot0, const SyncData<real> &self_dot1, SyncData<real> &dot_product, int m,
               int n,
               real gamma);

    void
    RBF_kernel(const SyncData<int> &self_dot0_idx, const SyncData<real> &self_dot1, SyncData<real> &dot_product, int m,
               int n, real gamma);

    void poly_kernel(SyncData<real> &dot_product, real gamma, real coef0, int degree, int mn);

    void sigmoid_kernel(SyncData<real> &dot_product, real gamma, real coef0, int mn);

    void sum_kernel_values(const SyncData<real> &coef, int total_sv, const SyncData<int> &sv_start,
                           const SyncData<int> &sv_count, const SyncData<real> &rho, const SyncData<real> &k_mat,
                           SyncData<real> &dec_values, int n_classes, int n_instances);

    void dns_csr_mul(int m, int n, int k, const SyncData<real> &dense_mat, const SyncData<real> &csr_val,
                     const SyncData<int> &csr_row_ptr, const SyncData<int> &csr_col_ind, int nnz,
                     SyncData<real> &result);
}
#endif //THUNDERSVM_KERNELMATRIX_KERNEL_H

