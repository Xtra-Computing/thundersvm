//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"

real rbf_kernel(const DataSet::node2d &instances, int x, int y, real gamma) {
    real res = 0;
    int i = 0;
    int j = 0;
    while (i < instances[x].size() || j < instances[y].size()) {
        if (j == instances[y].size() || instances[x][i].index < instances[y][j].index) {
            res += instances[x][i].value * instances[x][i].value;
            i++;
        } else if (i == instances[x].size() || instances[x][i].index > instances[y][j].index) {
            res += instances[y][j].value * instances[y][j].value;
            j++;
        } else {
            res += (instances[x][i].value - instances[y][j].value) * (instances[x][i].value - instances[y][j].value);
            i++;
            j++;
        }
    }
    return expf(-gamma * res);
}

TEST(KernelMatrixTest, get_rows) {
    DataSet dataSet;
//    dataSet.load_from_file("data/test_dataset.txt");
    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/a9a");
    real gamma = 0.5;
    KernelMatrix kernelMatrix(dataSet.instances(), dataSet.n_features(), gamma);
    int n_rows = 10;
    SyncData<int> rows(n_rows);
    SyncData<real> kernel_rows(n_rows * dataSet.total_count());
    for (int i = 0; i < n_rows; ++i) {
        rows.host_data()[i] = 2 + i;
    }
    rows.to_device();
    kernel_rows.to_device();
    kernelMatrix.get_rows(&rows, &kernel_rows);

    for (int i = 0; i < rows.count(); ++i) {
        for (int j = 0; j < kernelMatrix.m(); ++j) {
            real gpu_kernel = kernel_rows.host_data()[i * kernelMatrix.m() + j];
            real cpu_kernel = rbf_kernel(dataSet.instances(), rows.host_data()[i], j, gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << rows.host_data()[i] << j;
        }
    }

}
