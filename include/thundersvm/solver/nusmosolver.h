//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_NUSMOSOLVER_H
#define THUNDERSVM_NUSMOSOLVER_H

#include "csmosolver.h"

class NuSMOSolver : public CSMOSolver {
public:
    explicit NuSMOSolver(bool for_svr) : for_svr(for_svr) {};
protected:
    real
    calculate_rho(const SyncData<real> &f_val, const SyncData<int> &y, SyncData<real> &alpha, real Cp,
                  real Cn) const override;

    void smo_kernel(const SyncData<int> &y, SyncData<real> &f_val, SyncData<real> &alpha, SyncData<real> &alpha_diff,
                    const SyncData<int> &working_set, real Cp, real Cn, const SyncData<real> &k_mat_rows,
                    const SyncData<real> &k_mat_diag, int row_len, real eps, SyncData<real> &diff,
                    int max_iter) const override;

    void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                            const SyncData<real> &alpha, real Cp, real Cn, SyncData<int> &working_set) const override;

    void scale_alpha_rho(SyncData<real> &alpha, real &rho, real r) const;

private:
    bool for_svr;
};

#endif //THUNDERSVM_NUSMOSOLVER_H
