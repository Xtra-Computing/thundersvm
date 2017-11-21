//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_CSMOSOLVER_H
#define THUNDERSVM_CSMOSOLVER_H

#include <thundersvm/thundersvm.h>
#include <thundersvm/kernelmatrix.h>

/**
 * @brief C-SMO solver for SVC, SVR and OneClassSVC
 */
class CSMOSolver {
public:
    void solve(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<float_type> &alpha, float_type &rho,
               SyncData<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size) const;

protected:
    void init_f(const SyncData<float_type> &alpha, const SyncData<int> &y, const KernelMatrix &k_mat,
                SyncData<float_type> &f_val) const;

    virtual void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                                    const SyncData<float_type> &alpha, float_type Cp, float_type Cn,
                                    SyncData<int> &working_set) const;

    virtual float_type
    calculate_rho(const SyncData<float_type> &f_val, const SyncData<int> &y, SyncData<float_type> &alpha, float_type Cp,
                  float_type Cn) const;

    virtual void
    smo_kernel(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
               SyncData<float_type> &alpha_diff,
               const SyncData<int> &working_set, float_type Cp, float_type Cn, const SyncData<float_type> &k_mat_rows,
               const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
               int max_iter) const;
};

#endif //THUNDERSVM_CSMOSOLVER_H
