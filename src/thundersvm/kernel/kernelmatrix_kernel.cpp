//
// Created by jiashuai on 17-11-7.
//

#include <thundersvm/kernel/kernelmatrix_kernel.h>

#ifndef USE_CUDA

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace svm_kernel {
    void
    get_working_set_ins(const SyncData<float_type> &val, const SyncData<int> &col_ind, const SyncData<int> &row_ptr,
                        const SyncData<int> &data_row_idx, SyncData<float_type> &data_rows, int m) {
        const int *data_row_idx_data = data_row_idx.host_data();
        float_type *data_rows_data = data_rows.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const float_type *val_data = val.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            int row = data_row_idx_data[i];
            for (int j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
                int col = col_ind_data[j];
                data_rows_data[col * m + i] = val_data[j]; // row-major for cuSPARSE
            }
        }
    }

    void
    RBF_kernel(const SyncData<float_type> &self_dot0, const SyncData<float_type> &self_dot1,
               SyncData<float_type> &dot_product, int m,
               int n, float_type gamma) {
        float_type *dot_product_data = dot_product.host_data();
        const float_type *self_dot0_data = self_dot0.host_data();
        const float_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot0_data[i] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) * gamma);
            }
        }
    }

    void
    RBF_kernel(const SyncData<int> &self_dot0_idx, const SyncData<float_type> &self_dot1,
               SyncData<float_type> &dot_product, int m,
               int n, float_type gamma) {
        float_type *dot_product_data = dot_product.host_data();
        const int *self_dot0_idx_data = self_dot0_idx.host_data();
        const float_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) *
                        gamma);
            }
        }

    }

    void poly_kernel(SyncData<float_type> &dot_product, float_type gamma, float_type coef0, int degree, int mn) {
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
        }
    }

    void sigmoid_kernel(SyncData<float_type> &dot_product, float_type gamma, float_type coef0, int mn) {
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncData<float_type> &coef, int total_sv, const SyncData<int> &sv_start,
                           const SyncData<int> &sv_count, const SyncData<float_type> &rho,
                           const SyncData<float_type> &k_mat,
                           SyncData<float_type> &dec_values, int n_classes, int n_instances) {
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; idx++) {
            int k = 0;
            int n_binary_models = n_classes * (n_classes - 1) / 2;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    int si = sv_start[i];
                    int sj = sv_start[j];
                    int ci = sv_count[i];
                    int cj = sv_count[j];
                    const float_type *coef1 = &coef[(j - 1) * total_sv];
                    const float_type *coef2 = &coef[i * total_sv];
                    const float_type *k_values = &k_mat[idx * total_sv];
                    float_type sum = 0;
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < ci; ++l) {
                        sum += coef1[si + l] * k_values[si + l];
                    }
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < cj; ++l) {
                        sum += coef2[sj + l] * k_values[sj + l];
                    }
                    dec_values[idx * n_binary_models + k] = sum - rho[k];
                    k++;
                }
            }
        }
    }

    void dns_csr_mul(int m, int n, int k, const SyncData<float_type> &dense_mat, const SyncData<float_type> &csr_val,
                     const SyncData<int> &csr_row_ptr, const SyncData<int> &csr_col_ind, int nnz,
                     SyncData<float_type> &result) {
        /* 
        for(int row = 0; row < m; row ++){
            int nz_value_num = csr_row_ptr[row + 1] - csr_row_ptr[row];
            if(nz_value_num != 0){
                for(int col = 0; col < n; col++){
                    float_type sum = 0;
                    for(int nz_value_index = csr_row_ptr[row]; nz_value_index < csr_row_ptr[row + 1]; nz_value_index++){
                        sum += csr_val[nz_value_index] * dense_mat[col + csr_col_ind[nz_value_index] * n];
                    }
                    result[row * n + col] = sum;
                }
            }
        }
        */
        Eigen::Map<const Eigen::MatrixXf> denseMat(dense_mat.host_data(), n, k);
        Eigen::Map<const Eigen::SparseMatrix<float, Eigen::RowMajor>> sparseMat(m, k, nnz, csr_row_ptr.host_data(),
                                                                                csr_col_ind.host_data(),
                                                                                csr_val.host_data());
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dense_tran = denseMat.transpose();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat * dense_tran;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
                                                                                           retMat.rows(),
                                                                                           retMat.cols()) = retMat;

    }
}
#endif
