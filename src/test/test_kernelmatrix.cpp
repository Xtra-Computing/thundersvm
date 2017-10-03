//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"

real rbf_kernel(const DataSet::node2d &instances, int x, int y, real gamma) {
    real sum = 0;
    auto i = instances[x].begin();
    auto j = instances[y].begin();
    while (i != instances[x].end() && j != instances[y].end()) {
        if (i->index < j->index) {
            sum += i->value * i->value;
            i++;
        } else if (i->index > j->index) {
            sum += j->value * j->value;
            j++;
        } else {
            sum += (i->value - j->value) * (i->value - j->value);
            i++;
            j++;
        }
    }
    while (i != instances[x].end()) {
        sum += i->value * i->value;
        i++;
    }
    while (j != instances[y].end()) {
        sum += j->value * j->value;
        j++;
    }
    return expf(-gamma * sum);
}

TEST(KernelMatrixTest, get_rows) {
    DataSet dataSet;
//    dataSet.load_from_file("data/test_dataset.txt");
    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/a9a");
//    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/w8a");
    real gamma = 0.5;
    KernelMatrix kernelMatrix(dataSet.instances(), dataSet.n_features(), gamma);
    int n_rows = 10;
    SyncData<int> rows(n_rows);
    SyncData<real> kernel_rows(n_rows * dataSet.total_count());
    for (int i = 0; i < n_rows; ++i) {
        rows.host_data()[i] = i*3 + 4;
    }
    rows.to_device();
    kernel_rows.to_device();
    kernelMatrix.get_rows(rows, kernel_rows);

    for (int i = 0; i < rows.count(); ++i) {
        for (int j = 0; j < kernelMatrix.m(); ++j) {
            real gpu_kernel = kernel_rows.host_data()[i * kernelMatrix.m() + j];
            real cpu_kernel = rbf_kernel(dataSet.instances(), rows.host_data()[i], j, gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << rows.host_data()[i] << "," << j;
        }
    }

    DataSet::node2d instances(dataSet.instances().begin(), dataSet.instances().begin() + n_rows);
    kernelMatrix.get_rows(instances, kernel_rows);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < kernelMatrix.m(); ++j) {
            real gpu_kernel = kernel_rows.host_data()[i * kernelMatrix.m() + j];
            real cpu_kernel = rbf_kernel(dataSet.instances(), i, j, gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i << "," << j;
        }
    }
}
