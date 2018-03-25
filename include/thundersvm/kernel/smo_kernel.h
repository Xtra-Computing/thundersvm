//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SMO_KERNEL_H
#define THUNDERSVM_SMO_KERNEL_H

#include <thundersvm/thundersvm.h>
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncarray.h>

namespace svm_kernel {
    __host__ __device__ inline bool is_I_up(float_type a, float_type y, float_type Cp, float_type Cn) {
        return (y > 0 && a < Cp) || (y < 0 && a > 0);
    }

    __host__ __device__ inline bool is_I_low(float_type a, float_type y, float_type Cp, float_type Cn) {
        return (y > 0 && a > 0) || (y < 0 && a < Cn);
    }

    __host__ __device__ inline bool is_free(float_type a, float_type y, float_type Cp, float_type Cn) {
        return a > 0 && (y > 0 ? a < Cp : a < Cn);
    }

    void
    c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                SyncArray<float_type> &alpha_diff,
                const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                const SyncArray<kernel_type> &k_mat_rows,
                const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                int max_iter);

    void
    nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                 SyncArray<float_type> &alpha_diff,
                 const SyncArray<int> &working_set, float_type C, const SyncArray<kernel_type> &k_mat_rows,
                 const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                 int max_iter);

    void
    update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff, const SyncArray<kernel_type> &k_mat_rows,
             int n_instances);

    void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort);
}

#endif //THUNDERSVM_SMO_KERNEL_H
