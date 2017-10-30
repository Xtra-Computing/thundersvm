//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_CSMOSOLVER_H
#define THUNDERSVM_CSMOSOLVER_H

#include <thundersvm/thundersvm.h>
#include <thundersvm/kernelmatrix.h>

class CSMOSolver {
public:
    void solve(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
               SyncData<real> &f_val, real eps, real Cp, real Cn, int ws_size) const;

protected:
    void init_f(const SyncData<real> &alpha, const SyncData<int> &y, const KernelMatrix &k_mat,
                SyncData<real> &f_val) const;

    virtual void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                                    const SyncData<real> &alpha, real Cp, real Cn, SyncData<int> &working_set) const;

    virtual real
    calculate_rho(const SyncData<real> &f_val, const SyncData<int> &y, SyncData<real> &alpha, real Cp, real Cn) const;

    virtual void
    smo_kernel(const int *label, real *f_values, real *alpha, real *alpha_diff, const int *working_set,
               int ws_size, float Cp, float Cn, const float *k_mat_rows, const float *k_mat_diag,
               int row_len, real eps, real *diff_and_bias) const;
};

#endif //THUNDERSVM_CSMOSOLVER_H
