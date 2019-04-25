//
// Created by jiashuai on 17-10-30.
//
#include <thundersvm/model/nusvr.h>
#include <thundersvm/solver/nusmosolver.h>

void NuSVR::train(const DataSet &dataset, SvmParam param) {
    model_setup(dataset, param);
    int n_instances = dataset.n_instances();

    //duplicate instances
    DataSet::node2d instances_2(dataset.instances());
    instances_2.insert(instances_2.end(), dataset.instances().begin(), dataset.instances().end());

    KernelMatrix kernelMatrix(instances_2, param);

    SyncArray<float_type> f_val(n_instances * 2);
    SyncArray<int> y(n_instances * 2);

    SyncArray<float_type> alpha_2(n_instances * 2);
    float_type *f_val_data = f_val.host_data();
    int *y_data = y.host_data();
    float_type *alpha_2_data = alpha_2.host_data();
    float_type sum = param.C * param.nu * n_instances / 2;
    for (int i = 0; i < n_instances; ++i) {
        alpha_2_data[i] = alpha_2_data[i + n_instances] = min(sum, param.C);
        sum -= alpha_2_data[i];
        f_val_data[i] = f_val_data[i + n_instances] = -dataset.y()[i];
        y_data[i] = +1;
        y_data[i + n_instances] = -1;
    }

    int ws_size = get_working_set_size(n_instances * 2, kernelMatrix.n_features());
    NuSMOSolver solver(true);
    solver.solve(kernelMatrix, y, alpha_2, rho.host_data()[0], f_val, param.epsilon, param.C, param.C, ws_size, max_iter);
    save_svr_coef(alpha_2, dataset.instances());

    if(param.kernel_type == SvmParam::LINEAR){
        compute_linear_coef_single_model(dataset.n_features(), dataset.is_zero_based());
    }
}

void NuSVR::model_setup(const DataSet &dataset, SvmParam &param) {
    SVR::model_setup(dataset, param);
    this->param.svm_type = SvmParam::NU_SVR;
}
