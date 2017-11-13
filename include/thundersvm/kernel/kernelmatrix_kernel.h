//
// Created by jiashuai on 17-9-20.
//

#ifndef THUNDERSVM_KERNELMATRIX_KERNEL_H
#define THUNDERSVM_KERNELMATRIX_KERNEL_H

#include "thundersvm/thundersvm.h"
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncdata.h>

namespace svm_kernel {
    void
    get_working_set_ins(const SyncData<float_type> &val, const SyncData<int> &col_ind, const SyncData<int> &row_ptr,
                        const SyncData<int> &data_row_idx, SyncData<float_type> &data_rows, int m);

    void
    RBF_kernel(const SyncData<float_type> &self_dot0, const SyncData<float_type> &self_dot1,
               SyncData<float_type> &dot_product, int m,
               int n,
               float_type gamma);

    void
    RBF_kernel(const SyncData<int> &self_dot0_idx, const SyncData<float_type> &self_dot1,
               SyncData<float_type> &dot_product, int m,
               int n, float_type gamma);

    void poly_kernel(SyncData<float_type> &dot_product, float_type gamma, float_type coef0, int degree, int mn);

    void sigmoid_kernel(SyncData<float_type> &dot_product, float_type gamma, float_type coef0, int mn);

    void sum_kernel_values(const SyncData<float_type> &coef, int total_sv, const SyncData<int> &sv_start,
                           const SyncData<int> &sv_count, const SyncData<float_type> &rho,
                           const SyncData<float_type> &k_mat,
                           SyncData<float_type> &dec_values, int n_classes, int n_instances);

    void dns_csr_mul(int m, int n, int k, const SyncData<float_type> &dense_mat, const SyncData<float_type> &csr_val,
                     const SyncData<int> &csr_row_ptr, const SyncData<int> &csr_col_ind, int nnz,
                     SyncData<float_type> &result);
}
#endif //THUNDERSVM_KERNELMATRIX_KERNEL_H

