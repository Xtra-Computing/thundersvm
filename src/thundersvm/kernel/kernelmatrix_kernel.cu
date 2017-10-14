//
// Created by jiashuai on 17-9-20.
//
#include "thundersvm/kernel/kernelmatrix_kernel.h"

__global__ void
kernel_get_working_set_ins(const real *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                           real *data_rows,
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
    //m rows of kernel matrix, where m is the working set size; n is the number of training instances
    KERNEL_LOOP(idx, m * n) {
        int i = idx / n;//i is row id
        int j = idx % n;//j is column id
        dot_product[idx] = expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
    }
}

__global__ void
kernel_RBF_kernel(const int *self_dot0_idx, const real *self_dot1, real *dot_product, int m, int n, real gamma) {
    //compute m rows of kernel matrix, where m is the working set size and n is the number of training instances, according to idx
    KERNEL_LOOP(idx, m * n) {
        int i = idx / n;//i is row id
        int j = idx % n;//j is column id
        dot_product[idx] = expf(-(self_dot1[self_dot0_idx[i]] + self_dot1[j] - dot_product[idx] * 2) * gamma);
    }
}

__global__ void kernel_sum_kernel_values(const real *k_mat, int n_instances, int n_sv_unique, int n_bin_models,
                                         const int *sv_index, const real *coef, const int *sv_start,
                                         const int *sv_count,
                                         const real *rho, real *dec_values) {//compute decision values for n_instances
    KERNEL_LOOP(idx, n_instances * n_bin_models) {
        //one iteration uses a binary svm model to predict a decision value of an instance.
        int ins_id = idx / n_bin_models;
        int model_id = idx % n_bin_models;
        real sum = 0;
        const real *kernel_row = k_mat + ins_id * n_sv_unique;//kernel values of this instance
        int si = sv_start[model_id];
        int ci = sv_count[model_id];
        for (int i = 0; i < ci; ++i) {//TODO: improve by parallelism
            sum += coef[si + i] * kernel_row[sv_index[si + i]];//sv_index maps uncompressed sv idx to compressed sv idx.
        }
        dec_values[idx] = sum - rho[model_id];
    }
}

__global__ void kernel_poly_kernel(real *dot_product, real gamma, real coef0, int degree, int mn) {
    KERNEL_LOOP(idx, mn) {
        dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
    }
}

__global__ void kernel_sigmoid_kernel(real *dot_product, real gamma, real coef0, int mn) {
    KERNEL_LOOP(idx, mn) {
        dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
    }
}
