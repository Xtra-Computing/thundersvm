//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/syncdata.h>
#include <cusparse.h>
#include "thundersvm/kernel/kernelmatrix_kernel.h"

namespace svm_kernel {
    __global__ void
    kernel_get_working_set_ins(const float_type *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                               float_type *data_rows,
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
    kernel_RBF_kernel(const float_type *self_dot0, const float_type *self_dot1, float_type *dot_product, int m, int n,
                      float_type gamma) {
        //m rows of kernel matrix, where m is the working set size; n is the number of training instances
        KERNEL_LOOP(idx, m * n) {
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] = expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
        }
    }

    __global__ void
    kernel_RBF_kernel(const int *self_dot0_idx, const float_type *self_dot1, float_type *dot_product, int m, int n,
                      float_type gamma) {
        //compute m rows of kernel matrix, where m is the working set size and n is the number of training instances, according to idx
        KERNEL_LOOP(idx, m * n) {
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] = expf(-(self_dot1[self_dot0_idx[i]] + self_dot1[j] - dot_product[idx] * 2) * gamma);
        }
    }

    __global__ void
    kernel_sum_kernel_values(const float_type *coef, int total_sv, const int *sv_start, const int *sv_count,
                             const float_type *rho,
                             const float_type *k_mat, float_type *dec_values, int n_classes, int n_instances) {
        KERNEL_LOOP(idx, n_instances) {
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
                    for (int l = 0; l < ci; ++l) {
                        sum += coef1[si + l] * k_values[si + l];
                    }
                    for (int l = 0; l < cj; ++l) {
                        sum += coef2[sj + l] * k_values[sj + l];
                    }
                    dec_values[idx * n_binary_models + k] = sum - rho[k];
                    k++;
                }
            }
        }
    }

    __global__ void
    kernel_poly_kernel(float_type *dot_product, float_type gamma, float_type coef0, int degree, int mn) {
        KERNEL_LOOP(idx, mn) {
            dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
        }
    }

    __global__ void kernel_sigmoid_kernel(float_type *dot_product, float_type gamma, float_type coef0, int mn) {
        KERNEL_LOOP(idx, mn) {
            dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncData<float_type> &coef, int total_sv, const SyncData<int> &sv_start,
                           const SyncData<int> &sv_count, const SyncData<float_type> &rho,
                           const SyncData<float_type> &k_mat,
                           SyncData<float_type> &dec_values, int n_classes, int n_instances) {
        SAFE_KERNEL_LAUNCH(kernel_sum_kernel_values, coef.device_data(), total_sv, sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), k_mat.device_data(), dec_values.device_data(),
                           n_classes, n_instances);

    }

    void
    get_working_set_ins(const SyncData<float_type> &val, const SyncData<int> &col_ind, const SyncData<int> &row_ptr,
                        const SyncData<int> &data_row_idx, SyncData<float_type> &data_rows, int m) {
        SAFE_KERNEL_LAUNCH(kernel_get_working_set_ins, val.device_data(), col_ind.device_data(), row_ptr.device_data(),
                           data_row_idx.device_data(), data_rows.device_data(), m);

    }

    void
    RBF_kernel(const SyncData<float_type> &self_dot0, const SyncData<float_type> &self_dot1,
               SyncData<float_type> &dot_product, int m,
               int n,
               float_type gamma) {
        SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, self_dot0.device_data(), self_dot1.device_data(),
                           dot_product.device_data(), m, n, gamma);
    }

    void
    RBF_kernel(const SyncData<int> &self_dot0_idx, const SyncData<float_type> &self_dot1,
               SyncData<float_type> &dot_product, int m,
               int n, float_type gamma) {
        SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, self_dot0_idx.device_data(), self_dot1.device_data(),
                           dot_product.device_data(), m, n, gamma);
    }

    void poly_kernel(SyncData<float_type> &dot_product, float_type gamma, float_type coef0, int degree, int mn) {
        SAFE_KERNEL_LAUNCH(kernel_poly_kernel, dot_product.device_data(), gamma, coef0, degree, mn);
    }

    void sigmoid_kernel(SyncData<float_type> &dot_product, float_type gamma, float_type coef0, int mn) {
        SAFE_KERNEL_LAUNCH(kernel_sigmoid_kernel, dot_product.device_data(), gamma, coef0, mn);
    }

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    bool cusparse_init;

    void dns_csr_mul(int m, int n, int k, const SyncData<float_type> &dense_mat, const SyncData<float_type> &csr_val,
                     const SyncData<int> &csr_row_ptr, const SyncData<int> &csr_col_ind, int nnz,
                     SyncData<float_type> &result) {
        if (!cusparse_init) {
            cusparseCreate(&handle);
            cusparseCreateMatDescr(&descr);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparse_init = true;
        }
        float one(1);
        float zero(0);
        cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                        m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                        csr_col_ind.device_data(),
                        dense_mat.device_data(), n, &zero, result.device_data(), m);
        //cusparseScsrmm return row-major matrix, so no transpose is needed
    }
}
