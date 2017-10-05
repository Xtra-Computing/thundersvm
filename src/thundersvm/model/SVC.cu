//
// Created by jiashuai on 17-9-21.
//
#include <thundersvm/kernel/smo_kernel.h>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
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
            SyncData<int> y(ins.size());
            SyncData<real> alpha(ins.size());
            SyncData<real> init_f(ins.size());
            real rho;
            alpha.mem_set(0);
            for (int l = 0; l < dataSet.count()[i]; ++l) {
                y[l] = +1;
                init_f[l] = -1;
            }
            for (int l = 0; l < dataSet.count()[j]; ++l) {
                y[dataSet.count()[i] + l] = -1;
                init_f[dataSet.count()[i] + l] = +1;
            }
            KernelMatrix k_mat(ins, dataSet.n_features(), svmParam.gamma);
            int ws_size = min(max2power(dataSet.count()[0]), max2power(dataSet.count()[1])) * 2;
            smo_solver(k_mat, y, alpha, rho, init_f, 0.001, svmParam.C, ws_size);
            record_binary_model(k, alpha, y, rho, dataSet.original_index(i, j));
            k++;
        }
    }
}

vector<int> SVC::predict(const DataSet::node2d &instances, int batch_size) {
    //prepare device data
    SyncData<int> sv_start(n_binary_models);
    SyncData<int> sv_count(n_binary_models);
    int n_sv = 0;
    for (int i = 0; i < n_binary_models; ++i) {
        sv_start[i] = n_sv;
        sv_count[i] = this->coef[i].size();
        n_sv += this->coef[i].size();
    }
    SyncData<real> coef(n_sv);
    SyncData<int> sv_index(n_sv);
    SyncData<real> rho(n_binary_models);
    for (int i = 0; i < n_binary_models; ++i) {
        CUDA_CHECK(cudaMemcpy(coef.device_data() + sv_start[i], this->coef[i].data(), sizeof(real) * sv_count[i],
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sv_index.device_data() + sv_start[i], this->sv_index[i].data(), sizeof(int) * sv_count[i],
                              cudaMemcpyHostToDevice));
    }
    rho.copy_from(this->rho.data(), rho.count());

    //compute kernel values
    KernelMatrix k_mat(sv, dataSet.n_features(), svmParam.gamma);

    auto batch_start = instances.begin();
    auto batch_end = batch_start;
    vector<int> predict_y;
    while (batch_end != instances.end()) {
        while (batch_end != instances.end() && batch_end - batch_start < batch_size) batch_end++;
        DataSet::node2d batch_ins(batch_start, batch_end);
        SyncData<real> kernel_values(batch_ins.size() * sv.size());
        k_mat.get_rows(batch_ins, kernel_values);
        SyncData<real> dec_values(batch_ins.size() * n_binary_models);

        //sum kernel values and get decision values
        SAFE_KERNEL_LAUNCH(kernel_sum_kernel_values, kernel_values.device_data(), batch_ins.size(), sv.size(),
                           n_binary_models, sv_index.device_data(), coef.device_data(), sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), dec_values.device_data());

        //predict y by voting among k(k-1)/2 models
        for (int l = 0; l < batch_ins.size(); ++l) {
            vector<int> votes(n_binary_models, 0);
            int k = 0;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    if (dec_values[l * n_binary_models + k] > 0)
                        votes[i]++;
                    else
                        votes[j]++;
                    k++;
                }
            }
            int maxVoteClass = 0;
            for (int i = 0; i < n_classes; ++i) {
                if (votes[i] > votes[maxVoteClass])
                    maxVoteClass = i;
            }
            predict_y.push_back(dataSet.label()[maxVoteClass]);
        }
        batch_start += batch_size;
    }
    return predict_y;
}

void SVC::save_to_file(string path) {

}

void SVC::load_from_file(string path) {

}

void
SVC::smo_solver(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
                SyncData<real> &init_f, real eps, real C, int ws_size) {
    uint n_instances = k_mat.m();
    uint q = ws_size / 2;

    SyncData<int> working_set(ws_size);
    SyncData<int> working_set_first_half(q);
    SyncData<int> working_set_last_half(q);
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncData<real> f(n_instances);
    SyncData<int> f_idx(n_instances);
    SyncData<int> f_idx2sort(n_instances);
    SyncData<real> f_val2sort(n_instances);
    SyncData<real> alpha_diff(ws_size);
    SyncData<real> diff_and_bias(2);

    SyncData<real> k_mat_rows(ws_size * k_mat.m());
    SyncData<real> k_mat_rows_first_half(q * k_mat.m());
    SyncData<real> k_mat_rows_last_half(q * k_mat.m());
    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * k_mat.m()]);
    CHECK_EQ(init_f.count(), n_instances);
    f.copy_from(init_f);
    for (int i = 0; i < n_instances; ++i) {
        f_idx[i] = i;
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
            working_set_first_half.copy_from(working_set_last_half);
            ws = working_set_last_half.host_data();
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
        if (1 == iter) {
            k_mat.get_rows(working_set, k_mat_rows);
        } else {
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
            k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
        }

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
        SAFE_KERNEL_LAUNCH(update_f, f.device_data(), ws_size, alpha_diff.device_data(), k_mat_rows.device_data(),
                           n_instances);
    }
}

void SVC::record_binary_model(int k, const SyncData<real> &alpha, const SyncData<int> &y, real rho,
                              const vector<int> &original_index) {
    int n_sv = 0;
    for (int i = 0; i < alpha.count(); ++i) {
        if (alpha[i] != 0) {
            coef[k].push_back(alpha[i] * y[i]);
            if (sv_index_map.find(original_index[i]) == sv_index_map.end()) {
                int sv_index = sv_index_map.size();
                sv_index_map[original_index[i]] = sv_index;
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

int SVC::max2power(int n) const {
    return min(int(pow(2, floor(log2f(float(n))))), 512);
}

