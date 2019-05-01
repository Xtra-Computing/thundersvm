//
// Created by jiashuai on 17-10-6.
//

#include <iostream>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/solver/csmosolver.h>

void OneClassSVC::train(const DataSet &dataset, SvmParam param) {
    model_setup(dataset, param);
    int n_instances = dataset.n_instances();
    SyncArray<float_type> alpha(n_instances);
    SyncArray<float_type> f_val(n_instances);

    KernelMatrix kernelMatrix(dataset.instances(), param);

    alpha.mem_set(0);
    float_type *alpha_data = alpha.host_data();
    int n = static_cast<int>(param.nu * n_instances);
    for (int i = 0; i < n; ++i) {
        alpha_data[i] = 1;
    }
    if (n < n_instances)
        alpha_data[n] = param.nu * n_instances - n;
    int ws_size = get_working_set_size(n_instances, kernelMatrix.n_features());

    //TODO batch, thrust
    f_val.mem_set(0);
    SyncArray<int> y(n_instances);
    int *y_data = y.host_data();
    for (int i = 0; i < n_instances; ++i) {
        y_data[i] = 1;
    }
    CSMOSolver solver;
    solver.solve(kernelMatrix, y, alpha, rho.host_data()[0], f_val, param.epsilon, 1, 1, ws_size, max_iter);

    //todo these codes are similar to svr, try to combine them
    LOG(INFO) << "rho = " << rho.host_data()[0];
    vector<float_type> coef_vec;
    for (int i = 0; i < n_instances; ++i) {
        if (alpha_data[i] != 0) {
            sv.push_back(dataset.instances()[i]);
            sv_indices.push_back(i);
            coef_vec.push_back(alpha_data[i]);
        }
    }
    LOG(INFO) << "#sv = " << sv.size();
    n_sv.host_data()[0] = sv.size();
    n_sv.host_data()[1] = 0;
    n_total_sv = sv.size();
    coef.resize(coef_vec.size());
    coef.copy_from(coef_vec.data(), coef_vec.size());

    if(param.kernel_type == SvmParam::LINEAR){
        compute_linear_coef_single_model(dataset.n_features(), dataset.is_zero_based());
    }
}

vector<float_type> OneClassSVC::predict(const DataSet::node2d &instances, int batch_size) {
    vector<float_type> dec_values = SvmModel::predict(instances, batch_size);
    vector<float_type> predict_y;
    for (int i = 0; i < dec_values.size(); ++i) {
        predict_y.push_back(dec_values[i] > 0 ? 1 : -1);
    }
    return predict_y;
}


void OneClassSVC::model_setup(const DataSet &dataset, SvmParam &param) {
    SvmModel::model_setup(dataset, param);
    this->param.svm_type = SvmParam::ONE_CLASS;
}
