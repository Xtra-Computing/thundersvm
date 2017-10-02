//
// Created by jiashuai on 17-9-20.
//
#include "thundersvm/kernel/kernelmatrix_kernel.h"

__global__ void
kernel_get_data_rows(const real *val, const int *col_ind, const int *row_ptr, const int *data_row_idx, real *data_rows,
                     int m) {
    KERNEL_LOOP(i, m) {
        int row = data_row_idx[i];
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            int col = col_ind[j];
            data_rows[col * m + i] = val[j]; // row-major for cuSPARSE
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

__global__ void kernel_sum_kernel_values(const real *k_mat, int n_instances, int n_sv_unique, int n_bin_models,
                                         const int *sv_index, const real *coef, const int *sv_start,
                                         const int *sv_count,
                                         const real *rho, real *dec_values) {
    KERNEL_LOOP(idx, n_instances * n_bin_models) {
        int ins_id = idx / n_bin_models;
        int model_id = idx % n_bin_models;
        if (ins_id < n_instances) {
            real sum = 0;
            const real *kernel_row = k_mat + ins_id * n_sv_unique;//kernel values of this instance
            int si = sv_start[model_id];
            int ci = sv_count[model_id];
            for (int i = 0; i < ci; ++i) {//can be improved.
                sum += coef[si + i] * kernel_row[sv_index[si + i]];
            }
            dec_values[idx] = sum - rho[model_id];
        }
    }
}
