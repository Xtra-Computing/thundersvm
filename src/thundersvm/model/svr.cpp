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

    SyncArray<float_type> f_val(n_instances * 2);
    SyncArray<int> y(n_instances * 2);

    float_type *f_val_data = f_val.host_data();
    int *y_data = y.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_val_data[i] = param.p - dataset.y()[i];
        y_data[i] = +1;
        f_val_data[i + n_instances] = -param.p - dataset.y()[i];
        y_data[i + n_instances] = -1;
    }

    SyncArray<float_type> alpha_2(n_instances * 2);
    alpha_2.mem_set(0);
    int ws_size = get_working_set_size(n_instances * 2, kernelMatrix.n_features());
    CSMOSolver solver;
    solver.solve(kernelMatrix, y, alpha_2, rho.host_data()[0], f_val, param.epsilon, param.C, param.C, ws_size, max_iter);
    save_svr_coef(alpha_2, dataset.instances());

    if(param.kernel_type == SvmParam::LINEAR){
        compute_linear_coef_single_model(dataset.n_features(), dataset.is_zero_based());
    }
}

void SVR::save_svr_coef(const SyncArray<float_type> &alpha_2, const DataSet::node2d &instances) {
    LOG(INFO) << "rho = " << rho.host_data()[0];
    int n_instances = alpha_2.size() / 2;
    vector<float_type> coef_vec;
    const float_type *alpha_2_data = alpha_2.host_data();
    for (int i = 0; i < n_instances; ++i) {
        float_type alpha_i = alpha_2_data[i] - alpha_2_data[i + n_instances];
        if (alpha_i != 0) {
            sv.push_back(instances[i]);
            sv_indices.push_back(i);
            coef_vec.push_back(alpha_i);
        }
    }
    LOG(INFO) << "#sv = " << sv.size();
    n_sv.host_data()[0] = sv.size();
    n_sv.host_data()[1] = 0;
    n_total_sv = sv.size();
    coef.resize(coef_vec.size());
    coef.copy_from(coef_vec.data(), coef_vec.size());
}

void SVR::model_setup(const DataSet &dataset, SvmParam &param) {
    SvmModel::model_setup(dataset, param);
    this->param.svm_type = SvmParam::EPSILON_SVR;
}


