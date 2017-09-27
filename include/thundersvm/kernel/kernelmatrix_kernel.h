//
// Created by jiashuai on 17-9-20.
//

#ifndef THUNDERSVM_KERNELMATRIX_KERNEL_H
#define THUNDERSVM_KERNELMATRIX_KERNEL_H

#include "thundersvm/thundersvm.h"
#include "thundersvm/clion_cuda.h"

__global__ void
kernel_get_data_rows(const real *val, const int *col_ind, const int *row_ptr, const int *data_row_idx, real *data_rows,
                     int m, int n);

__global__ void
kernel_RBF_kernel(const real *self_dot0, const real *self_dot1, real *dot_product, int m, int n, real gamma);

__global__ void
kernel_RBF_kernel(const int *self_dot0_idx, const real *self_dot1, real *dot_product, int m, int n, real gamma);

#endif //THUNDERSVM_KERNELMATRIX_KERNEL_H
