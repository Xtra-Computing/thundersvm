//
// Created by jiashuai on 17-9-21.
//
#include <thundersvm/kernel/smo_kernel.h>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <thundersvm/model/svc.h>
#include "thrust/sort.h"

SVC::SVC(DataSet &dataSet, const SvmParam &svmParam) : SvmModel(dataSet, svmParam) {
    dataSet.group_classes();
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
            DataSet::node2d ins = dataSet.instances(i, j);//get instances of class i and j
            SyncData<int> y(ins.size());
            SyncData<real> alpha(ins.size());
            SyncData<real> f(ins.size());
            real rho;
            alpha.mem_set(0);
            for (int l = 0; l < dataSet.count()[i]; ++l) {
                y[l] = +1;
                f[l] = -1;
            }
            for (int l = 0; l < dataSet.count()[j]; ++l) {
                y[dataSet.count()[i] + l] = -1;
                f[dataSet.count()[i] + l] = +1;
            }
            KernelMatrix k_mat(ins, dataSet.n_features(), svmParam.gamma);
            int ws_size = min(min(max2power(dataSet.count()[0]), max2power(dataSet.count()[1])) * 2, 1024);
            smo_solver(k_mat, y, alpha, rho, f, 0.001, svmParam.C, ws_size);//TODO: use eps in svm_param
            record_binary_model(k, alpha, y, rho, dataSet.original_index(i, j));
            k++;
        }
    }
}

vector<real> SVC::predict(const DataSet::node2d &instances, int batch_size) {
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
    KernelMatrix k_mat(sv, dataSet.n_features(), svmParam.gamma);//TODO: initialize dataSet in svm_load.

    auto batch_start = instances.begin();
    auto batch_end = batch_start;
    vector<real> predict_y;
    while (batch_end != instances.end()) {
        while (batch_end != instances.end() && batch_end - batch_start < batch_size) batch_end++;

        DataSet::node2d batch_ins(batch_start, batch_end);//get a batch of instances
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
            predict_y.push_back((float) dataSet.label()[maxVoteClass]);
        }
        batch_start += batch_size;
    }
    return predict_y;
}

void SVC::save_to_file(string path) {

}

void SVC::load_from_file(string path) {

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
            sv_index[k].push_back(sv_index_map[original_index[i]]);//save unique sv id.
            n_sv++;
        }
    }
    this->rho[k] = rho;
    LOG(INFO) << "rho=" << rho;
    LOG(INFO) << "#SV=" << n_sv;
}


