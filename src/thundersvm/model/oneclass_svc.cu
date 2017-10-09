//
// Created by jiashuai on 17-10-6.
//

#include <thundersvm/model/oneclass_svc.h>

OneClassSVC::OneClassSVC(DataSet &dataSet, const SvmParam &svmParam) : SvmModel(dataSet, svmParam) {}

void OneClassSVC::train() {
    SyncData<real> alpha(n_instances);
    SyncData<real> f(n_instances);

    KernelMatrix kernelMatrix(dataSet.instances(), dataSet.n_features(), svmParam.gamma);

    alpha.mem_set(0);
    int n = static_cast<int>(svmParam.nu * n_instances);
    for (int i = 0; i < n; ++i) {
        alpha[i] = 1;
    }
    if (n < n_instances)
        alpha[n] = svmParam.nu * n_instances - n;
    int ws_size = min(min(max2power(n), max2power(n_instances - n)) * 2, 1024);

    //init_f
    //TODO batch, thrust
    SyncData<int> idx(1);
    SyncData<real> kernel_row(n_instances);
    f.mem_set(0);
    for (int i = 0; i <= n; ++i) {
        idx[0] = i;
        kernelMatrix.get_rows(idx, kernel_row);
        for (int j = 0; j < n_instances; ++j) {
            f[i] += alpha[i] * kernel_row[j];
        }
    }

    SyncData<int> y(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        y[i] = 1;
    }
    smo_solver(kernelMatrix, y, alpha, rho, f, 0.001, 1, ws_size);
    record_model(alpha, y);
}

void OneClassSVC::save_to_file(string path) {

}

void OneClassSVC::load_from_file(string path) {

}
