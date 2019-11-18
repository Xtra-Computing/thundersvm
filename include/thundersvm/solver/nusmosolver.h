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
    ~NuSMOSolver() override = default;
protected:
    float_type
    calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                  float_type Cp,
                  float_type Cn) const override;

    void smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                    SyncArray<float_type> &alpha_diff,
                    const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                    const SyncArray<kernel_type> &k_mat_rows,
                    const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                    int max_iter) const override;

    void select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                            const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                            SyncArray<int> &working_set) const override;

    void scale_alpha_rho(SyncArray<float_type> &alpha, float_type &rho, float_type r) const;

private:
    bool for_svr;
};

#endif //THUNDERSVM_NUSMOSOLVER_H
