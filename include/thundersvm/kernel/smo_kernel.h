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
    c_smo_solve(const SyncData<int> &y, SyncData<real> &f_val, SyncData<real> &alpha, SyncData<real> &alpha_diff,
                const SyncData<int> &working_set, real Cp, real Cn, const SyncData<real> &k_mat_rows,
                const SyncData<real> &k_mat_diag, int row_len, real eps, SyncData<real> &diff, int max_iter);

    void
    nu_smo_solve(const SyncData<int> &y, SyncData<real> &f_val, SyncData<real> &alpha, SyncData<real> &alpha_diff,
                 const SyncData<int> &working_set, real C, const SyncData<real> &k_mat_rows,
                 const SyncData<real> &k_mat_diag, int row_len, real eps, SyncData<real> &diff, int max_iter);

    void
    update_f(SyncData<real> &f, const SyncData<real> &alpha_diff, const SyncData<real> &k_mat_rows, int n_instances);

    void sort_f(SyncData<real> &f_val2sort, SyncData<int> &f_idx2sort);
}

#endif //THUNDERSVM_SMO_KERNEL_H
