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
    int ws_size = min(max2power(ins.size()), 1024);
    NuSMOSolver solver(false);
    solver.solve(k_mat, y, alpha, rho, f_val, param.epsilon, 1, 1, ws_size);

    //todo these codes are identical to svc
    LOG(INFO)<<"rho = "<<rho;
    int n_sv = 0;
    for (int l = 0; l < alpha.size(); ++l) {
        alpha[l] *= y[l];
        if (alpha[l] != 0) n_sv++;
    }
    LOG(INFO)<<"#sv = "<<n_sv;
}

