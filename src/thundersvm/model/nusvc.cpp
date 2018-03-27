//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/model/nusvc.h>
#include <thundersvm/solver/nusmosolver.h>

void NuSVC::train_binary(const DataSet &dataset, int i, int j, SyncArray<float_type> &alpha, float_type &rho) {
    DataSet::node2d ins = dataset.instances(i, j);//get instances of class i and j
    int n_pos = dataset.count()[i];
    int n_neg = dataset.count()[j];
    SyncArray<int> y(ins.size());
    alpha.resize(ins.size());
    SyncArray<float_type> f_val(ins.size());
    alpha.mem_set(0);
    f_val.mem_set(0);
    float_type sum_pos = param.nu * ins.size() / 2;
    float_type sum_neg = sum_pos;
    int *y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int l = 0; l < n_pos; ++l) {
        y_data[l] = +1;
        alpha_data[l] = min(1., sum_pos);
        sum_pos -= alpha_data[l];
    }
    for (int l = 0; l < n_neg; ++l) {
        y_data[n_pos + l] = -1;
        alpha_data[n_pos + l] = min(1., sum_neg);
        sum_neg -= alpha_data[n_pos + l];
    }
    vector<int> ori = dataset.original_index(i, j);

    KernelMatrix k_mat(ins, param);
    int ws_size = get_working_set_size(ins.size(), k_mat.n_features());
    NuSMOSolver solver(false);
    solver.solve(k_mat, y, alpha, rho, f_val, param.epsilon, 1, 1, ws_size, max_iter);

    LOG(INFO)<<"rho = "<<rho;
    int n_sv = 0;
    alpha_data = alpha.host_data();
    for (int l = 0; l < alpha.size(); ++l) {
        alpha_data[l] *= y_data[l];
        if (alpha_data[l] != 0) n_sv++;
    }
    LOG(INFO)<<"#sv = "<<n_sv;
}

void NuSVC::model_setup(const DataSet &dataset, SvmParam &param) {
    SVC::model_setup(dataset, param);
    this->param.svm_type = SvmParam::NU_SVC;
}
