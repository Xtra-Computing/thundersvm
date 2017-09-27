//
// Created by jiashuai on 17-9-20.
//
#include "thundersvm/kernel/kernelmatrix_kernel.h"

__global__ void
kernel_get_data_rows(const real *val, const int *col_ind, const int *row_ptr, const int *data_row_idx, real *data_rows,
                     int m, int n) {
    KERNEL_LOOP(i, m) {
        int row = data_row_idx[i];
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            int col = col_ind[j];
            data_rows[i * n + col] = val[j];
        }
    }
}

__global__ void
kernel_RBF_kernel(const real *self_dot0, const real *self_dot1, real *dot_product, int m, int n, real gamma) {
    KERNEL_LOOP(idx, m * n) {
        int i = idx / n;
        int j = idx % n;
        dot_product[idx] = expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
    }
}

__global__ void
kernel_RBF_kernel(const int *self_dot0_idx, const real *self_dot1, real *dot_product, int m, int n, real gamma) {
    KERNEL_LOOP(idx, m * n) {
        int i = idx / n;
        int j = idx % n;
        dot_product[idx] = expf(-(self_dot1[self_dot0_idx[i]] + self_dot1[j] - dot_product[idx] * 2) * gamma);
    }
}
