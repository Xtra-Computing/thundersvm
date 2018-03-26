//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/nusmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>

using namespace svm_kernel;

float_type
NuSMOSolver::calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                           float_type Cp,
                           float_type Cn) const {
    int n_free_p = 0, n_free_n = 0;
    float_type sum_free_p = 0, sum_free_n = 0;
    float_type up_value_p = INFINITY, up_value_n = INFINITY;
    float_type low_value_p = -INFINITY, low_value_n = -INFINITY;
    const float_type *f_val_data = f_val.host_data();
    const int *y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (y_data[i] > 0) {
            if (alpha_data[i] > 0 && alpha_data[i] < Cp) {
                n_free_p++;
                sum_free_p += f_val_data[i];
            }
            if (is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value_p = min(up_value_p, -f_val_data[i]);
            if (is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value_p = max(low_value_p, -f_val_data[i]);
        } else {
            if (alpha_data[i] > 0 && alpha_data[i] < Cn) {
                n_free_n++;
                sum_free_n += -f_val_data[i];
            }
            if (is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value_n = min(up_value_n, -f_val_data[i]);
            if (is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value_n = max(low_value_n, -f_val_data[i]);
        }
    }
    float_type r1 = n_free_p != 0 ? sum_free_p / n_free_p : (-(up_value_p + low_value_p) / 2);
    float_type r2 = n_free_n != 0 ? sum_free_n / n_free_n : (-(up_value_n + low_value_n) / 2);
    float_type rho = (r1 - r2) / 2;
    //not scale for svr, scale for nu-svc
    if (!for_svr) {
        float_type r = (r1 + r2) / 2;
        scale_alpha_rho(alpha, rho, r);
    }
    return rho;
}

void
NuSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                                const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                                SyncArray<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left_p = 0;
    int p_left_n = 0;
    int p_right_p = n_instances - 1;
    int p_right_n = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    int *working_set_data = working_set.host_data();
    int ws_size = working_set.size();
    while (n_selected < ws_size) {
        int i;
        if (p_left_p < n_instances) {
            i = index[p_left_p];
            while (ws_indicator[i] == 1 || !(y_data[i] > 0 && is_I_up(alpha_data[i], y_data[i], Cp, Cn))) {
                //construct working set of I_up
                p_left_p++;
                if (p_left_p == n_instances) break;
                i = index[p_left_p];
            }
            if (p_left_p < n_instances) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_left_n < n_instances) {
            i = index[p_left_n];
            while (ws_indicator[i] == 1 || !(y_data[i] < 0 && is_I_up(alpha_data[i], y_data[i], Cp, Cn))) {
                //construct working set of I_up
                p_left_n++;
                if (p_left_n == n_instances) break;
                i = index[p_left_n];
            }
            if (p_left_n < n_instances && n_selected < ws_size) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right_p >= 0) {
            i = index[p_right_p];
            while (ws_indicator[i] == 1 || !(y_data[i] > 0 && is_I_low(alpha_data[i], y_data[i], Cp, Cn))) {
                //construct working set of I_low
                p_right_p--;
                if (p_right_p == -1) break;
                i = index[p_right_p];
            }
            if (p_right_p >= 0 && n_selected < ws_size) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right_n >= 0) {
            i = index[p_right_n];
            while (ws_indicator[i] == 1 || !(y_data[i] < 0 && is_I_low(alpha_data[i], y_data[i], Cp, Cn))) {
                //construct working set of I_low
                p_right_n--;
                if (p_right_n == -1) break;
                i = index[p_right_n];
            }
            if (p_right_n >= 0 && n_selected < ws_size) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
    }
}

void NuSMOSolver::scale_alpha_rho(SyncArray<float_type> &alpha, float_type &rho, float_type r) const {
    float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        alpha_data[i] /= r;//TODO parallel
    }
    rho /= r;
}

void NuSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                             SyncArray<float_type> &alpha_diff, const SyncArray<int> &working_set, float_type Cp,
                             float_type Cn,
                             const SyncArray<kernel_type> &k_mat_rows, const SyncArray<kernel_type> &k_mat_diag,
                             int row_len, float_type eps,
                             SyncArray<float_type> &diff, int max_iter) const {
    //Cn is not used but for compatibility with c-svc
    nu_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}
