//
// Created by jiashuai on 17-11-7.
//

#include <thundersvm/kernel/smo_kernel.h>

namespace svm_kernel {
    void c_smo_solve(const SyncData<int> &y, SyncData<real> &f_val, SyncData<real> &alpha, SyncData<real> &alpha_diff,
                     const SyncData<int> &working_set, real Cp, real Cn, const SyncData<real> &k_mat_rows,
                     const SyncData<real> &k_mat_diag, int row_len, real eps, SyncData<real> &diff, int max_iter) {

    }

    void nu_smo_solve(const SyncData<int> &y, SyncData<real> &f_val, SyncData<real> &alpha, SyncData<real> &alpha_diff,
                      const SyncData<int> &working_set, real C, const SyncData<real> &k_mat_rows,
                      const SyncData<real> &k_mat_diag, int row_len, real eps, SyncData<real> &diff, int max_iter) {

    }

    void
    update_f(SyncData<real> &f, const SyncData<real> &alpha_diff, const SyncData<real> &k_mat_rows, int n_instances) {

    }

    void sort_f(SyncData<real> &f_val2sort, SyncData<int> &f_idx2sort) {

    }
}
