//
// Created by jiashuai on 17-10-5.
//
#include <iostream>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <thundersvm/solver/csmosolver.h>
#include "thundersvm/model/svr.h"

void SVR::train(const DataSet &dataset, SvmParam param) {
    model_setup(dataset, param);

    int n_instances = dataset.n_instances();
    //duplicate instances
    DataSet::node2d instances_2(dataset.instances());
    instances_2.insert(instances_2.end(), dataset.instances().begin(), dataset.instances().end());

    KernelMatrix kernelMatrix(instances_2, param);

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
    CSMOSolver solver;
    solver.solve(kernelMatrix, y, alpha_2, rho[0], f_val, param.epsilon, param.C, param.C, ws_size);
    save_svr_coef(alpha_2, dataset.instances());
}

void SVR::save_svr_coef(const SyncData<real> &alpha_2, const DataSet::node2d &instances) {
    LOG(INFO) << "rho = " << rho[0];
    int n_instances = alpha_2.size() / 2;
    vector<real> coef_vec;
    for (int i = 0; i < n_instances; ++i) {
        real alpha_i = alpha_2[i] - alpha_2[i + n_instances];
        if (alpha_i != 0) {
            sv.push_back(instances[i]);
            coef_vec.push_back(alpha_i);
        }
    }
    LOG(INFO) << "#sv = " << sv.size();
    n_sv[0] = sv.size();
    n_sv[1] = 0;
    coef.resize(coef_vec.size());
    coef.copy_from(coef_vec.data(), coef_vec.size());
}

