//
// Created by jiashuai on 17-10-5.
//
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include "thundersvm/model/svr.h"

void SVR::train(DataSet dataset, SvmParam param) {
    int n_instances = dataset.total_count();

    //duplicate instances
    DataSet::node2d instances_2(dataset.instances());
    instances_2.insert(instances_2.end(), dataset.instances().begin(), dataset.instances().end());

    KernelMatrix kernelMatrix(instances_2, param.gamma);

    SyncData<real> f_val(n_instances * 2);
    SyncData<int> y(n_instances * 2);

    for (int i = 0; i < n_instances; ++i) {
        f_val[i] = param.p - dataset.y()[i];
        y[i] = +1;
        f_val[i + n_instances] = -param.p - dataset.y()[i];
        y[i + n_instances] = -1;
    }

    SyncData<real> alpha_2(n_instances * 2);
    alpha_2.mem_set(0);
    int ws_size = min(max2power(n_instances) * 2, 1024);
    smo_solver(kernelMatrix, y, alpha_2, rho, f_val, param.epsilon, param.C, ws_size);
    SyncData<real> alpha(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        alpha[i] = alpha_2[i] - alpha_2[i + n_instances];
    }
    record_model(alpha, y, dataset.instances(), param);
}


void SVR::save_to_file(string path) {

}

void SVR::load_from_file(string path) {

}




