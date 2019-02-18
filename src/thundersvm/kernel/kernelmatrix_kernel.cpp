//
// Created by jiashuai on 17-11-7.
//

#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
namespace svm_kernel {
    void
    get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                            const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n) {
        const int *data_row_idx_data = data_row_idx.host_data();
        kernel_type *data_rows_data = data_rows.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const kernel_type *val_data = val.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            int row = data_row_idx_data[i];
            for (int j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
                int col = col_ind_data[j];
                data_rows_data[i * n + col] = val_data[j]; //row major
            }
        }
    }

    void
    get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                        const SyncArray<int> &data_row_idx, SyncArray <kernel_type>& ws_val,
                        SyncArray<int> &ws_col_ind, SyncArray<int> &ws_row_ptr, int m){
        const int *data_row_idx_data = data_row_idx.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const kernel_type *val_data = val.host_data();
//        kernel_type *ws_val_data = ws_val.host_data();
//        int *ws_row_ptr_data = ws_row_ptr.host_data();
//        int *ws_col_ind_data = ws_col_ind.host_data();
        //three arrays for csr representation
        vector<kernel_type> csr_val;
        vector<int> csr_col_ind;//index of each value of all the instances
        vector<int> csr_row_ptr(1, 0);//the start positions of the instances
        //ws_row_ptr_data[0] = 0;
        for(int i = 0; i < m; i++){
            int row = data_row_idx_data[i];
            for(int j = row_ptr_data[row]; j < row_ptr_data[row+1]; ++j){
                csr_col_ind.push_back(col_ind_data[j]);
                csr_val.push_back(val_data[j]);
            }
            csr_row_ptr.push_back(csr_row_ptr.back() + row_ptr_data[row+1] - row_ptr_data[row]);
        }
        //three arrays (on GPU/CPU) for csr representation
        ws_val.resize(csr_val.size());
        ws_col_ind.resize(csr_col_ind.size());
        ws_row_ptr.resize(csr_row_ptr.size());
        //copy data to the three arrays
        ws_val.copy_from(csr_val.data(), ws_val.size());
        ws_col_ind.copy_from(csr_col_ind.data(), ws_col_ind.size());
        ws_row_ptr.copy_from(csr_row_ptr.data(), ws_row_ptr.size());
    }

    void
    RBF_kernel(const SyncArray<kernel_type> &self_dot0, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        kernel_type *dot_product_data = dot_product.host_data();
        const kernel_type *self_dot0_data = self_dot0.host_data();
        const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot0_data[i] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) * gamma);
            }
        }
    }

    void
    RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        kernel_type *dot_product_data = dot_product.host_data();
        const int *self_dot0_idx_data = self_dot0_idx.host_data();
        const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) *
                        gamma);
            }
        }
    }

    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product_data[idx] = powf(gamma * dot_product_data[idx] + coef0, degree);
        }
    }

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product_data[idx] = tanhf(gamma * dot_product_data[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances) {
        const int *sv_start_data = sv_start.host_data();
        const int *sv_count_data = sv_count.host_data();
        const float_type *coef_data = coef.host_data();
        const kernel_type *k_mat_data = k_mat.host_data();
        float_type *dec_values_data = dec_values.host_data();
        const float_type* rho_data = rho.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; idx++) {
            int k = 0;
            int n_binary_models = n_classes * (n_classes - 1) / 2;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    int si = sv_start_data[i];
                    int sj = sv_start_data[j];
                    int ci = sv_count_data[i];
                    int cj = sv_count_data[j];
                    const float_type *coef1 = &coef_data[(j - 1) * total_sv];
                    const float_type *coef2 = &coef_data[i * total_sv];
                    const kernel_type *k_values = &k_mat_data[idx * total_sv];
                    double sum = 0;
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < ci; ++l) {
                        sum += coef1[si + l] * k_values[si + l];
                    }
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < cj; ++l) {
                        sum += coef2[sj + l] * k_values[sj + l];
                    }
                    dec_values_data[idx * n_binary_models + k] = sum - rho_data[k];
                    k++;
                }
            }
        }
    }

    void dns_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result) {
        Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> denseMat(dense_mat.host_data(), k, n);
        Eigen::Map<const Eigen::SparseMatrix<kernel_type, Eigen::RowMajor>> sparseMat(m, k, nnz, csr_row_ptr.host_data(),
                                                                                csr_col_ind.host_data(),
                                                                                csr_val.host_data());
        Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat * denseMat;
        Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
                                                                                           retMat.rows(),
                                                                                           retMat.cols()) = retMat;
    }

    void csr_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &ws_val, const SyncArray<int> &ws_col_ind,
                     const SyncArray<int> &ws_row_ptr, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz, int nnz2,
                     SyncArray<kernel_type> &result) {
        Eigen::Map<const Eigen::SparseMatrix<kernel_type, Eigen::RowMajor>> sparseMat1(m, k, nnz, csr_row_ptr.host_data(),
                                                                                      csr_col_ind.host_data(),
                                                                                      csr_val.host_data());
        Eigen::Map<const Eigen::SparseMatrix<kernel_type>> sparseMat2(k, n, nnz2, ws_row_ptr.host_data(),
                                                                      ws_col_ind.host_data(),
                                                                      ws_val.host_data());
        Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat1 * sparseMat2;
        Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
                                                                                                 retMat.rows(),
                                                                                                 retMat.cols()) = retMat;
    }

    void dns_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat,
                     const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result){
        Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> denseMat(dense_mat.host_data(), k, n);
        Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                originDenseMat(origin_dense.host_data(), m, k);
        Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = originDenseMat * denseMat;
        Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
                                                                                                 retMat.rows(),
                                                                                                 retMat.cols()) = retMat;
    }
}
