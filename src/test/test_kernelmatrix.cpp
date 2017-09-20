//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"
TEST(KernelMatrixTest, get_rows){
    DataSet dataSet;
    dataSet.load_from_file("data/test_dataset.txt");
    KernelMatrix kernelMatrix(dataSet.index(), dataSet.value(), dataSet.n_features(), 1);
    int n_rows = 10;
    SyncData<int> rows(n_rows);
    SyncData<real> kernel_rows(n_rows * dataSet.total_count());
    for (int i = 0; i < n_rows; ++i) {
        rows.host_data()[i] = 2 + i * 3;
    }
    rows.to_device();
    kernel_rows.to_device();
    kernelMatrix.get_rows(&rows, &kernel_rows);

    //check diagonal element equals to one
    for (int i = 0; i < n_rows; ++i) {
        EXPECT_NEAR(kernel_rows.host_data()[i * dataSet.total_count() + rows.host_data()[i]],1, 1e-5);
    }
}
