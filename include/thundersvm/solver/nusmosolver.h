//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_NUSMOSOLVER_H
#define THUNDERSVM_NUSMOSOLVER_H

#include "csmosolver.h"

class NuSMOSolver : public CSMOSolver {
protected:
    real
    calculate_rho(const SyncData<real> &f_val, const SyncData<int> &y, SyncData<real> &alpha, real C) const override;

    void
    smo_kernel(const int *label, real *f_values, real *alpha, real *alpha_diff, const int *working_set, int ws_size,
               float C, const float *k_mat_rows, const float *k_mat_diag, int row_len, real eps,
               real *diff_and_bias) const override;

    void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                            const SyncData<real> &alpha, real C, SyncData<int> &working_set) const override;

};
#endif //THUNDERSVM_NUSMOSOLVER_H
