//
// Created by jiashuai on 17-11-7.
//
#include <thundersvm/kernel/kernelmatrix_kernel.h>
//#include <mkl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
namespace svm_kernel {
    void get_working_set_ins(const SyncData<real> &val, const SyncData<int> &col_ind, const SyncData<int> &row_ptr,
                             const SyncData<int> &data_row_idx, SyncData<real> &data_rows, int m) {
        //std::cout << "val[0]" << val[0] << std::endl;
        //std::cout << "val.host_data[0]" << val.host_data()[0] <<std::endl;
        #pragma omp parallel for
        for(int i = 0; i < m; i++) {
            int row = data_row_idx[i];
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                int col = col_ind[j];
                data_rows[col * m + i] = val[j]; // row-major for cuSPARSE
            }
        }
        //std::cout << "data_rows[0]" << data_rows[0] << std::endl;
        //std::cout << "data_rows.host_data()[0]" << data_rows.host_data()[0] << std::endl;
    }

    void
    RBF_kernel(const SyncData<real> &self_dot0, const SyncData<real> &self_dot1, SyncData<real> &dot_product, int m,
               int n, real gamma) {
        #pragma omp parallel for
        for(int idx = 0; idx < m * n; idx++){
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] = expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
        }
    }

    void
    RBF_kernel(const SyncData<int> &self_dot0_idx, const SyncData<real> &self_dot1, SyncData<real> &dot_product, int m,
               int n, real gamma) {
        #pragma omp parallel for
        for(int idx = 0; idx < m * n; idx++){
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] = expf(-(self_dot1[self_dot0_idx[i]] + self_dot1[j] - dot_product[idx] * 2) * gamma);
        }

    }

    void poly_kernel(SyncData<real> &dot_product, real gamma, real coef0, int degree, int mn) {
        #pragma omp parallel for
        for(int idx = 0; idx < mn; idx++){
            dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
        }
    }

    void sigmoid_kernel(SyncData<real> &dot_product, real gamma, real coef0, int mn) {
        #pragma omp parallel for
        for(int idx = 0; idx < mn; idx++){
            dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncData<real> &coef, int total_sv, const SyncData<int> &sv_start,
                           const SyncData<int> &sv_count, const SyncData<real> &rho, const SyncData<real> &k_mat,
                           SyncData<real> &dec_values, int n_classes, int n_instances) {
        #pragma omp parallel for
        for(int idx = 0; idx < n_instances; idx++){
            int k = 0;
            int n_binary_models = n_classes * (n_classes - 1) / 2;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    int si = sv_start[i];
                    int sj = sv_start[j];
                    int ci = sv_count[i];
                    int cj = sv_count[j];
                    const real *coef1 = &coef[(j - 1) * total_sv];
                    const real *coef2 = &coef[i * total_sv];
                    const real *k_values = &k_mat[idx * total_sv];
                    real sum = 0;
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
    
    void dns_csr_mul(int m, int n, int k, const SyncData<real> &dense_mat, const SyncData<real> &csr_val,
                     const SyncData<int> &csr_row_ptr, const SyncData<int> &csr_col_ind, int nnz,
                     SyncData<real> &result) {
        /*
        for(int row = 0; row < m; row ++){
            int nz_value_num = csr_row_ptr[row + 1] - csr_row_ptr[row];
            if(nz_value_num != 0){
                for(int col = 0; col < n; col++){
                    real sum = 0;
                    for(int nz_value_index = csr_row_ptr[row]; nz_value_index < csr_row_ptr[row + 1]; nz_value_index++){
                        sum += csr_val[nz_value_index] * dense_mat[col + csr_col_ind[nz_value_index] * n];
                    }
                    result[row * n + col] = sum;
                }
            }
        }
        */
        /*
        Eigen::Map<Eigen::Matrix<real, n, k, Eigen::ColMajor> > denseMat(dense_mat.host_data());
        Eigen::Map<Eigen::SparseMatrix<real, Eigen::RowMajor> > sparseMat(m, k, nnz, csr_row_ptr.host_data(), csr_col_ind.host_data(), csr_val.host_data());
        Eigen::Matrix<real, n, m, Eigen::RowMajor> retMat = denseMat * sparseMat.transpose();
        Eigen::Map<Marix<real, n, m, Eigen::RowMajor> > (result.host_data(), retMat.rows(), retMat.cols()) = retMat;    
        */
        Eigen::Map<const Eigen::MatrixXf> denseMat(dense_mat.host_data(), n, k);
        Eigen::Map<const Eigen::SparseMatrix<float, Eigen::RowMajor>> sparseMat(m, k, nnz, csr_row_ptr.host_data(), csr_col_ind.host_data(), csr_val.host_data());
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> retMat = denseMat * sparseMat.transpose();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > (result.host_data(), retMat.rows(), retMat.cols()) = retMat; 
        /*
        float one(1);
        float zero(0);
        char matdescra[4] = {'G', 0, 0, 'C'};
        char transa = 'N';
        const char* matdescra_ptr = &matdescra[0];
        //dense_mat transpose
        mkl_simatcopy();
        const int* m_ptr = &m;
        const int* n_ptr = &n;
        const int* k_ptr = &k;
        //BLAS_usmm
        mkl_scsrmm(&transa, m_ptr, n_ptr, k_ptr, &one, matdescra_ptr, csr_val.host_data(), csr_col_ind.host_data(), 
            csr_row_ptr.host_data(), &csr_row_ptr.host_data()[1], dense_mat.host_data(), n_ptr, &zero, result.host_data(), m_ptr); 
        */
    }
}
