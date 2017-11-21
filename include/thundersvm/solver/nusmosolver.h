//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_NUSMOSOLVER_H
#define THUNDERSVM_NUSMOSOLVER_H

#include "csmosolver.h"

/**
 * @brief Nu-SMO solver for NuSVC, NuSVR
 */
class NuSMOSolver : public CSMOSolver {
public:
    explicit NuSMOSolver(bool for_svr) : for_svr(for_svr) {};
protected:
    float_type
    calculate_rho(const SyncData<float_type> &f_val, const SyncData<int> &y, SyncData<float_type> &alpha, float_type Cp,
                  float_type Cn) const override;

    void smo_kernel(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                    SyncData<float_type> &alpha_diff,
                    const SyncData<int> &working_set, float_type Cp, float_type Cn,
                    const SyncData<float_type> &k_mat_rows,
                    const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                    int max_iter) const override;

    void select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                            const SyncData<float_type> &alpha, float_type Cp, float_type Cn,
                            SyncData<int> &working_set) const override;

    void scale_alpha_rho(SyncData<float_type> &alpha, float_type &rho, float_type r) const;

private:
    bool for_svr;
};

#endif //THUNDERSVM_NUSMOSOLVER_H
