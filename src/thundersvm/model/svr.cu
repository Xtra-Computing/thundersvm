//
// Created by jiashuai on 17-10-5.
//
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include "thundersvm/model/svr.h"

SVR::SVR(DataSet &dataSet, const SvmParam &svmParam) : SvmModel(dataSet, svmParam) {
    n_instances = dataSet.instances().size();
}

void SVR::train() {
    //duplicate instances
    DataSet::node2d instances_2 = dataSet.instances();
    instances_2.insert(instances_2.end(), dataSet.instances().begin(), dataSet.instances().end());

    KernelMatrix kernelMatrix(instances_2, dataSet.n_features(), svmParam.gamma);

    SyncData<real> init_f(n_instances * 2);
    SyncData<int> y(n_instances * 2);
    for (int i = 0; i < n_instances; ++i) {
        init_f[i] = svmParam.p - dataSet.y()[i];
        y[i] = +1;
        init_f[i + n_instances] = -svmParam.p - dataSet.y()[i];
        y[i + n_instances] = -1;
    }

    SyncData<real> alpha(n_instances * 2);
    real rho;
    int ws_size = min(max2power(n_instances) * 2, 1024);
    smo_solver(kernelMatrix, y, alpha, rho, init_f, 0.001, svmParam.C, ws_size);
    record_model(alpha, y, rho);
}

vector<real> SVR::predict(const DataSet::node2d &instances, int batch_size) {
    //prepare device data
    int n_sv = coef.size();
    SyncData<real> coef(n_sv);
    SyncData<int> sv_index(n_sv);
    SyncData<int> sv_start(1);
    SyncData<int> sv_count(1);
    SyncData<real> rho(1);

    sv_start[0] = 0;
    sv_count[0] = n_sv;
    rho[0] = this->rho;
    coef.copy_from(this->coef.data(), n_sv);
    sv_index.copy_from(this->sv_index.data(), n_sv);

    //compute kernel values
    KernelMatrix k_mat(sv, dataSet.n_features(), svmParam.gamma);

    auto batch_start = instances.begin();
    auto batch_end = batch_start;
    vector<real> predict_y;
    while (batch_end != instances.end()) {
        while (batch_end != instances.end() && batch_end - batch_start < batch_size) batch_end++;
        DataSet::node2d batch_ins(batch_start, batch_end);
        SyncData<real> kernel_values(batch_ins.size() * sv.size());
        k_mat.get_rows(batch_ins, kernel_values);
        SyncData<real> dec_values(batch_ins.size());

        //sum kernel values and get decision values
        SAFE_KERNEL_LAUNCH(kernel_sum_kernel_values, kernel_values.device_data(), batch_ins.size(), sv.size(),
                           1, sv_index.device_data(), coef.device_data(), sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), dec_values.device_data());

        for (int i = 0; i < batch_ins.size(); ++i) {
            predict_y.push_back(dec_values[i]);
        }
        batch_start += batch_size;
    }
    return predict_y;
}

void SVR::save_to_file(string path) {

}

void SVR::load_from_file(string path) {

}

void SVR::record_model(const SyncData<real> &alpha, const SyncData<int> &y, real rho) {
    int n_sv = 0;
    for (int i = 0; i < n_instances * 2; ++i) {
        if (alpha[i] != 0) {
            coef.push_back(alpha[i] * y[i]);
            int original_index = i > n_instances ? (i - n_instances) : i;
            if (sv_map.find(original_index) == sv_map.end()) {
                int sv_index = sv_map.size();
                sv_map[original_index] = sv_index;
                sv.push_back(dataSet.instances()[original_index]);
            }
            sv_index.push_back(sv_map[original_index]);
            n_sv++;
        }
    }
    this->rho = rho;
    LOG(INFO) << "RHO = " << rho;
    LOG(INFO) << "#SV = " << n_sv;
}

