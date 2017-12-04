//
// Created by jiashuai on 17-9-20.
//

#ifndef THUNDERSVM_KERNELMATRIX_KERNEL_H
#define THUNDERSVM_KERNELMATRIX_KERNEL_H

#include "thundersvm/thundersvm.h"
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncarray.h>

namespace svm_kernel {
    void
    get_working_set_ins(const SyncArray<float_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                        const SyncArray<int> &data_row_idx, SyncArray<float_type> &data_rows, int m);

    void
    RBF_kernel(const SyncArray<float_type> &self_dot0, const SyncArray<float_type> &self_dot1,
               SyncArray<float_type> &dot_product, int m,
               int n,
               float_type gamma);

    void
    RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<float_type> &self_dot1,
               SyncArray<float_type> &dot_product, int m,
               int n, float_type gamma);

    void poly_kernel(SyncArray<float_type> &dot_product, float_type gamma, float_type coef0, int degree, int mn);

    void sigmoid_kernel(SyncArray<float_type> &dot_product, float_type gamma, float_type coef0, int mn);

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<float_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances);

    void dns_csr_mul(int m, int n, int k, const SyncArray<float_type> &dense_mat, const SyncArray<float_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<float_type> &result);
}
#endif //THUNDERSVM_KERNELMATRIX_KERNEL_H

