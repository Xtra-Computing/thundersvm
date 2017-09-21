//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"

real rbf_kernel(const vector<vector<int>> &index, const vector<vector<real>> &value, int x, int y, real gamma) {
    real res = 0;
    int i = 0;
    int j = 0;
    while (i < index[x].size() || j < index[y].size()) {
        if (j == index[y].size() || index[x][i] < index[y][j]) {
            res += value[x][i] * value[x][i];
            i++;
        } else if (i == index[x].size() || index[x][i] > index[y][j]){
            res += value[y][j] * value[y][j];
            j++;
        } else {
            res += (value[x][i] - value[y][j]) * (value[x][i] - value[y][j]);
            i++;
            j++;
        }
    }
    return expf(-gamma * res);
}

TEST(KernelMatrixTest, get_rows) {
    DataSet dataSet;
    dataSet.load_from_file("data/test_dataset.txt");
    real gamma = 1;
    KernelMatrix kernelMatrix(dataSet.index(), dataSet.value(), dataSet.n_features(), gamma);
    int n_rows = 10;
    SyncData<int> rows(n_rows);
    SyncData<real> kernel_rows(n_rows * dataSet.total_count());
    for (int i = 0; i < n_rows; ++i) {
        rows.host_data()[i] = 2 + i;
    }
    rows.to_device();
    kernel_rows.to_device();
    kernelMatrix.get_rows(&rows, &kernel_rows);

    //check diagonal element equals to one
    for (int i = 0; i < rows.count(); ++i) {
        for (int j = 0; j < kernelMatrix.m(); ++j) {
            real gpu_kernel = kernel_rows.host_data()[i * kernelMatrix.m() + j];
            real cpu_kernel = rbf_kernel(dataSet.index(), dataSet.value(), rows.host_data()[i], j, gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5)<<rows.host_data()[i]<<j;
        }
    }

}
