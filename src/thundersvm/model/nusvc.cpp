//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/model/nusvc.h>
#include <thundersvm/solver/nusmosolver.h>

void NuSVC::train_binary(const DataSet &dataset, int i, int j, int k) {
    DataSet::node2d ins = dataset.instances(i, j);//get instances of class i and j
    int n_pos = dataset.count()[i];
    int n_neg = dataset.count()[j];
    SyncData<int> y(ins.size());
    SyncData<real> alpha(ins.size());
    SyncData<real> f_val(ins.size());
    real rho;
    alpha.mem_set(0);
    f_val.mem_set(0);
    real sum_pos = param.nu * ins.size() / 2;
    real sum_neg = sum_pos;
    for (int l = 0; l < n_pos; ++l) {
        y[l] = +1;
        alpha[l] = min(1.f, sum_pos);
        sum_pos -= alpha[l];
    }
    for (int l = 0; l < n_neg; ++l) {
        y[n_pos + l] = -1;
        alpha[n_pos + l] = min(1.f, sum_neg);
        sum_neg -= alpha[n_pos + l];
    }
    vector<int> ori = dataset.original_index(i, j);

    KernelMatrix k_mat(ins, param);
    int ws_size = min(min(max2power(dataset.count()[i]), max2power(dataset.count()[j])) * 2, 1024);
    NuSMOSolver solver(false);
    solver.solve(k_mat, y, alpha, rho, f_val, param.epsilon, 1, 1, ws_size);
    record_binary_model(k, alpha, y, rho, dataset.original_index(i, j), dataset.instances());
}

