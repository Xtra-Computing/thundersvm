//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SMO_KERNEL_H
#define THUNDERSVM_SMO_KERNEL_H

#include <thundersvm/thundersvm.h>
#include <thundersvm/clion_cuda.h>
#include <thundersvm/syncdata.h>

namespace svm_kernel {
    __host__ __device__ inline bool is_I_up(float a, float y, float Cp, float Cn) {
        return (y > 0 && a < Cp) || (y < 0 && a > 0);
    }

    __host__ __device__ inline bool is_I_low(float a, float y, float Cp, float Cn) {
        return (y > 0 && a > 0) || (y < 0 && a < Cn);
    }

    __host__ __device__ inline bool is_free(float a, float y, float Cp, float Cn) {
        return a > 0 && (y > 0 ? a < Cp : a < Cn);
    }

    void
    c_smo_solve(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                SyncData<float_type> &alpha_diff,
                const SyncData<int> &working_set, float_type Cp, float_type Cn, const SyncData<float_type> &k_mat_rows,
                const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                int max_iter);

    void
    nu_smo_solve(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                 SyncData<float_type> &alpha_diff,
                 const SyncData<int> &working_set, float_type C, const SyncData<float_type> &k_mat_rows,
                 const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                 int max_iter);

    void
    update_f(SyncData<float_type> &f, const SyncData<float_type> &alpha_diff, const SyncData<float_type> &k_mat_rows,
             int n_instances);

    void sort_f(SyncData<float_type> &f_val2sort, SyncData<int> &f_idx2sort);
}

#endif //THUNDERSVM_SMO_KERNEL_H
