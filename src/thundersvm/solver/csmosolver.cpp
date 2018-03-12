//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/csmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>
#include <limits.h>

using namespace svm_kernel;

void
CSMOSolver::solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
                  SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size, int out_max_iter) const {
    int n_instances = k_mat.n_instances();
    int q = ws_size / 2;

    SyncArray<int> working_set(ws_size);
    SyncArray<int> working_set_first_half(q);
    SyncArray<int> working_set_last_half(q);
#ifdef USE_CUDA
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
#endif
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncArray<int> f_idx(n_instances);
    SyncArray<int> f_idx2sort(n_instances);
    SyncArray<float_type> f_val2sort(n_instances);
    SyncArray<float_type> alpha_diff(ws_size);
    SyncArray<float_type> diff(2);

    SyncArray<float_type> k_mat_rows(ws_size * k_mat.n_instances());
    SyncArray<float_type> k_mat_rows_first_half(q * k_mat.n_instances());
    SyncArray<float_type> k_mat_rows_last_half(q * k_mat.n_instances());
#ifdef USE_CUDA
    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * k_mat.n_instances()]);
#else
    k_mat_rows_first_half.set_host_data(k_mat_rows.host_data());
    k_mat_rows_last_half.set_host_data(&k_mat_rows.host_data()[q * k_mat.n_instances()]);
#endif
    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG(INFO) << "training start";
    int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
    long long local_iter = 0;
    for (int iter = 0;; ++iter) {
        //select working set
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
        sort_f(f_val2sort, f_idx2sort);
        vector<int> ws_indicator(n_instances, 0);
        if (0 == iter) {
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set);
            k_mat.get_rows(working_set, k_mat_rows);
        } else {
            working_set_first_half.copy_from(working_set_last_half);
            int *working_set_data = working_set.host_data();
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set_data[i]] = 1;
            }
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set_last_half);
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
            k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
        }
        //local smo
        smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps, diff,
                   max_iter);
        //update f
        update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances());
        if (iter % 100 == 0) LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter;
        local_iter += diff.host_data()[1];
        if (diff.host_data()[0] < eps) {
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
            LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter;
            break;
        }
        if ((out_max_iter != -1) && (iter == out_max_iter)){
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
            LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter;
            break;
        }
    }
}

void
CSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                               const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                               SyncArray<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    int *working_set_data = working_set.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !is_I_up(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_up
                p_left++;
                if (p_left == n_instances) break;
                i = index[p_left];
            }
            if (p_left < n_instances) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !is_I_low(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }

    }
}

float_type
CSMOSolver::calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                          float_type Cp,
                          float_type Cn) const {
    int n_free = 0;
    double sum_free = 0;
    float_type up_value = INFINITY;
    float_type low_value = -INFINITY;
    const float_type *f_val_data = f_val.host_data();
    const int *y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (is_free(alpha_data[i], y_data[i], Cp, Cn)) {
            n_free++;
            sum_free += f_val_data[i];
        }
        if (is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value = min(up_value, f_val_data[i]);
        if (is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value = max(low_value, f_val_data[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void CSMOSolver::init_f(const SyncArray<float_type> &alpha, const SyncArray<int> &y, const KernelMatrix &k_mat,
                        SyncArray<float_type> &f_val) const {
    //todo auto set batch size
    int batch_size = 100;
    vector<int> idx_vec;
    vector<float_type> alpha_diff_vec;
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha_data[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha_data[i] * y_data[i]);
        }
        if (idx_vec.size() > batch_size || (i == alpha.size() - 1 && idx_vec.size() > 0)) {
            SyncArray<int> idx(idx_vec.size());
            SyncArray<float_type> alpha_diff(idx_vec.size());
            idx.copy_from(idx_vec.data(), idx_vec.size());
            alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
            SyncArray<float_type> kernel_rows(idx.size() * k_mat.n_instances());
            k_mat.get_rows(idx, kernel_rows);
            update_f(f_val, alpha_diff, kernel_rows, k_mat.n_instances());
            idx_vec.clear();
            alpha_diff_vec.clear();
        }
    }
}

void
CSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                       SyncArray<float_type> &alpha_diff,
                       const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                       const SyncArray<float_type> &k_mat_rows,
                       const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps,
                       SyncArray<float_type> &diff,
                       int max_iter) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}

