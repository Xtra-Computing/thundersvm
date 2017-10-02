//
// Created by jiashuai on 17-9-21.
//
#include <thundersvm/kernel/smo_kernel.h>
#include "thundersvm/model/SVC.h"
#include "thrust/sort.h"
#include "thrust/system/cuda/execution_policy.h"

SVC::SVC(DataSet &dataSet, const SvmParam &svmParam) : SvmModel(dataSet, svmParam) {
    n_classes = dataSet.n_classes();
    n_binary_models = n_classes * (n_classes - 1) / 2;
    rho.resize(n_binary_models);
    sv_index.resize(n_binary_models);
    coef.resize(n_binary_models);
}

void SVC::train() {
    int k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            DataSet::node2d ins = dataSet.instances(i, j);
            size_t subproblem_size = ins.size();
            SyncData<int> y(subproblem_size);
            SyncData<real> alpha(subproblem_size);
            real rho;
            alpha.mem_set(0);
            for (int l = 0; l < dataSet.count()[i]; ++l) {
                y[l] = +1;
            }
            for (int l = 0; l < dataSet.count()[j]; ++l) {
                y[dataSet.count()[i] + l] = -1;
            }
            KernelMatrix k_mat(ins, dataSet.n_features(), svmParam.gamma);
            smo_solver(k_mat, y, alpha, rho, 0.001, svmParam.C);
            record_binary_model(k, alpha, y, rho, dataSet.original_index(i, j));
            k++;
        }
    }
}

void SVC::predict(DataSet &dataSet) {

}

void SVC::save_to_file(string path) {

}

void SVC::load_from_file(string path) {

}

void SVC::smo_solver(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho, real eps,
                     real C) {
//    TIMED_FUNC(timer_obj);
    uint n_instances = k_mat.m();
    SyncData<real> f(n_instances);
//    LOG(INFO)<<min(10.,ceil(log2(float(n_instances))));
    uint ws_size = 1024;
    uint q = ws_size / 2;
    SyncData<int> working_set(ws_size);
    SyncData<int> f_idx(n_instances);
    SyncData<int> f_idx2sort(n_instances);
    SyncData<real> f_val2sort(n_instances);
    SyncData<real> alpha_diff(ws_size);
    SyncData<real> k_mat_rows(ws_size * k_mat.m());
    SyncData<real> diff_and_bias(2);
    for (int i = 0; i < n_instances; ++i) {
        f.host_data()[i] = -y.host_data()[i];
        f_idx.host_data()[i] = i;
    }
    alpha.mem_set(0);
    LOG(INFO) << "training start";
    for (int iter = 1;; ++iter) {
        //select working set
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f);
        thrust::sort_by_key(thrust::cuda::par, f_val2sort.device_data(), f_val2sort.device_data() + n_instances,
                            f_idx2sort.device_data(), thrust::less<real>());
        int *ws;
        vector<int> ws_indicator(n_instances, 0);
        if (1 == iter) {
            ws = working_set.host_data();
            q = ws_size;
        } else {
            q = ws_size / 2;
            working_set.copy_from(working_set.device_data() + q, q);
            ws = working_set.host_data() + q;
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set[i]] = 1;
            }
        }
        int p_left = 0;
        int p_right = n_instances - 1;
        int n_selected = 0;
        const int *index = f_idx2sort.host_data();
        while (n_selected < q) {
            int i;
            if (p_left < n_instances) {
                i = index[p_left];
                while (ws_indicator[i] == 1 || !(y[i] > 0 && alpha[i] < C || y[i] < 0 && alpha[i] > 0)) {
                    p_left++;
                    if (p_left == n_instances) break;
                    i = index[p_left];
                }
                if (p_left < n_instances) {
                    ws[n_selected++] = i;
                    ws_indicator[i] = 1;
                }
            }
            if (p_right >= 0) {
                i = index[p_right];
                while ((ws_indicator[i] == 1 || !(y[i] > 0 && alpha[i] > 0 || y[i] < 0 && alpha[i] < C))) {
                    p_right--;
                    if (p_right == -1) break;
                    i = index[p_right];
                }
                if (p_right >= 0) {
                    ws[n_selected++] = i;
                    ws_indicator[i] = 1;
                }
            }
        }

        //precompute kernel
        working_set.to_device();
        k_mat.get_rows(&working_set, &k_mat_rows);
        //local smo
        size_t smem_size = ws_size * sizeof(real) * 3 + 2 * sizeof(float);
        localSMO << < 1, ws_size, smem_size >> >
                                  (y.device_data(), f.device_data(), alpha.device_data(), alpha_diff.device_data(),
                                          working_set.device_data(), ws_size, C, k_mat_rows.device_data(), n_instances,
                                          eps, diff_and_bias.device_data());
        LOG_EVERY_N(10, INFO) << "diff=" << diff_and_bias[0];
        if (diff_and_bias[0] < eps) {
            rho = diff_and_bias[1];
            break;
        }
        //update f
        update_f << < NUM_BLOCKS, BLOCK_SIZE >> >
                                  (f.device_data(), ws_size, alpha_diff.device_data(), k_mat_rows.device_data(), n_instances);
    }
}

void SVC::record_binary_model(int k, const SyncData<real> &alpha, const SyncData<int> &y, real rho,
                              const vector<int> &original_index) {
    int n_sv = 0;
    for (int i = 0; i < alpha.count(); ++i) {
        if (alpha[i] != 0) {
            coef[k].push_back(alpha[i] * y[i]);
            if (sv_index_map.find(original_index[i]) == sv_index_map.end()) {
                sv_index_map[original_index[i]] = sv_index_map.size();
                sv.push_back(dataSet.instances()[original_index[i]]);
            }
            sv_index[k].push_back(sv_index_map[original_index[i]]);
            n_sv++;
        }
    }
    this->rho[k] = rho;
    LOG(INFO) << "rho=" << rho;
    LOG(INFO) << "#SV=" << n_sv;
}

