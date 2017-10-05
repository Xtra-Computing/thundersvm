//
// Created by jiashuai on 17-10-5.
//
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
            sv.push_back(dataSet.instances()[i > n_instances ? i - n_instances : i]);
            n_sv++;
        }
    }
    this->rho = rho;
    LOG(INFO) << "RHO = " << rho;
    LOG(INFO) << "#SV = " << n_sv;
}

