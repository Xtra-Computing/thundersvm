//
// Created by jiashuai on 17-10-5.
//
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include "thundersvm/model/svr.h"

SVR::SVR(DataSet &dataSet, const SvmParam &svmParam) : SvmModel(dataSet, svmParam) {
}

void SVR::train() {
    //duplicate instances
    DataSet::node2d instances_2(dataSet.instances());
    instances_2.insert(instances_2.end(), dataSet.instances().begin(), dataSet.instances().end());

    KernelMatrix kernelMatrix(instances_2, dataSet.n_features(), svmParam.gamma);

    SyncData<real> f(n_instances * 2);
    SyncData<int> y(n_instances * 2);

    for (int i = 0; i < n_instances; ++i) {
        f[i] = svmParam.p - dataSet.y()[i];
        y[i] = +1;
        f[i + n_instances] = -svmParam.p - dataSet.y()[i];
        y[i + n_instances] = -1;
    }

    SyncData<real> alpha(n_instances * 2);
    alpha.mem_set(0);
    int ws_size = min(max2power(n_instances) * 2, 1024);
    smo_solver(kernelMatrix, y, alpha, rho, f, 0.001, svmParam.C, ws_size);//TODO: use eps in svm_param
    for (int i = 0; i < n_instances; ++i) {
        alpha[i] -= alpha[i + n_instances];
    }
    record_model(alpha, y);
}


void SVR::save_to_file(string path) {

}

void SVR::load_from_file(string path) {

}


