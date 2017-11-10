//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/csmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>
#include <limits.h>
//#include <iostream>
//using namespace std;

using namespace svm_kernel;
void CSMOSolver::solve(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
                       SyncData<real> &f_val, real eps, real Cp, real Cn, int ws_size) const {
    uint n_instances = k_mat.n_instances();
    uint q = ws_size / 2;

    SyncData<int> working_set(ws_size);
    SyncData<int> working_set_first_half(q);
    SyncData<int> working_set_last_half(q);
#ifdef USE_CUDA
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
#endif
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncData<int> f_idx(n_instances);
    SyncData<int> f_idx2sort(n_instances);
    SyncData<real> f_val2sort(n_instances);
    SyncData<real> alpha_diff(ws_size);
    SyncData<real> diff(1);

    SyncData<real> k_mat_rows(ws_size * k_mat.n_instances());
    SyncData<real> k_mat_rows_first_half(q * k_mat.n_instances());
    SyncData<real> k_mat_rows_last_half(q * k_mat.n_instances());
#ifdef USE_CUDA
    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * k_mat.n_instances()]);
#else
    k_mat_rows_first_half.set_host_data(k_mat_rows.host_data());
    k_mat_rows_last_half.set_host_data(&k_mat_rows.host_data()[q * k_mat.n_instances()]);
#endif
    for (int i = 0; i < n_instances; ++i) {
        f_idx[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG(INFO) << "training start";
    int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
    for (int iter = 0;; ++iter) {
        //select working set
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
        sort_f(f_val2sort, f_idx2sort);
        vector<int> ws_indicator(n_instances, 0);
        if (0 == iter) {
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set);
            k_mat.get_rows(working_set, k_mat_rows);
	    //cout<<"iter = 1"<<endl;
       	    //cout<<"kmat[0]"<<k_mat_rows[0]<<endl;
	    } else {
	    //cout<<"iter != 1"<<endl;
            working_set_first_half.copy_from(working_set_last_half);
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set[i]] = 1;
            }
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set_last_half);
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
            k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
            //cout<<"kmatlast[0]"<<k_mat_rows_last_half[0]<<endl;
	    }
        //local smo
        smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps, diff,
                   max_iter);
        //update f
        update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances());
        if (iter % 10 == 0) {
            printf(".");
            std::cout.flush();
        }
        if (diff[0] < eps) {
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
            break;
        }
    }
    printf("\n");
}

void CSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                                    const SyncData<real> &alpha, real Cp, real Cn, SyncData<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !is_I_up(alpha[i], y[i], Cp, Cn)) {
                //construct working set of I_up
                p_left++;
                if (p_left == n_instances) break;
                i = index[p_left];
            }
            if (p_left < n_instances) {
                working_set[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !is_I_low(alpha[i], y[i], Cp, Cn)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }

    }
}

real
CSMOSolver::calculate_rho(const SyncData<real> &f_val, const SyncData<int> &y, SyncData<real> &alpha, real Cp,
                          real Cn) const {
    int n_free = 0;
    real sum_free = 0;
    real up_value = INFINITY;
    real low_value = -INFINITY;
    for (int i = 0; i < alpha.size(); ++i) {
        if (is_free(alpha[i], y[i], Cp, Cn)) {
            n_free++;
            sum_free += f_val[i];
        }
        if (is_I_up(alpha[i], y[i], Cp, Cn)) up_value = min(up_value, f_val[i]);
        if (is_I_low(alpha[i], y[i], Cp, Cn)) low_value = max(low_value, f_val[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void CSMOSolver::init_f(const SyncData<real> &alpha, const SyncData<int> &y, const KernelMatrix &k_mat,
                        SyncData<real> &f_val) const {
    //todo auto set batch size
    int batch_size = 100;
    vector<int> idx_vec;
    vector<real> alpha_diff_vec;
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha[i] * y[i]);
        }
        if (idx_vec.size() > batch_size || (i == alpha.size() - 1 && idx_vec.size() > 0)) {
            SyncData<int> idx(idx_vec.size());
            SyncData<real> alpha_diff(idx_vec.size());
            idx.copy_from(idx_vec.data(), idx_vec.size());
            alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
            SyncData<real> kernel_rows(idx.size() * k_mat.n_instances());
            k_mat.get_rows(idx, kernel_rows);
            update_f(f_val, alpha_diff, kernel_rows, k_mat.n_instances());
            idx_vec.clear();
            alpha_diff_vec.clear();
        }
    }
}

void
CSMOSolver::smo_kernel(const SyncData<int> &y, SyncData<real> &f_val, SyncData<real> &alpha, SyncData<real> &alpha_diff,
                       const SyncData<int> &working_set, real Cp, real Cn, const SyncData<real> &k_mat_rows,
                       const SyncData<real> &k_mat_diag, int row_len, real eps, SyncData<real> &diff,
                       int max_iter) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}

