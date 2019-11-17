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
    void solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
               SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size, int max_iter) const;

    virtual ~CSMOSolver() = default;

protected:
    void init_f(const SyncArray<float_type> &alpha, const SyncArray<int> &y, const KernelMatrix &k_mat,
                SyncArray<float_type> &f_val) const;

    virtual void
    select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                       const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                       SyncArray<int> &working_set) const;

    virtual float_type
    calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                  float_type Cp,
                  float_type Cn) const;
    float_type calculate_obj(const SyncArray<float_type> &f_val, const SyncArray<float_type> &alpha,
                                 const SyncArray<int> &y) const;

    virtual void
    smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
               SyncArray<float_type> &alpha_diff,
               const SyncArray<int> &working_set, float_type Cp, float_type Cn, const SyncArray<kernel_type> &k_mat_rows,
               const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
               int max_iter) const;
};

#endif //THUNDERSVM_CSMOSOLVER_H
