//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/syncarray.h>
#include <cusparse.h>
#include "thundersvm/kernel/kernelmatrix_kernel.h"
#include <thundersvm/config.h>

#include <cub/cub.cuh>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()
extern long long time1;
extern long long time3;

using namespace cub;
namespace svm_kernel {
    CachingDeviceAllocator  g_allocator(2, 3, 11, CachingDeviceAllocator::INVALID_SIZE, true, false); 
    __global__ void
    kernel_get_working_set_ins(const kernel_type *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                               kernel_type *data_rows,
                               int m, int n) {
        KERNEL_LOOP(i, m) {
            int row = data_row_idx[i];
            for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
                int col = col_ind[j];
                data_rows[col * m + i] = val[j]; // col-major for cuSPARSE
            }
            //replace col-major to row-major
            //int row = data_row_idx[i];
            //for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            //    int col = col_ind[j];
            //    data_rows[i*n+col] = val[j]; // row-major for cuSPARSE
            //}
        }
    }

    __global__ void
    kernel_RBF_kernel(const kernel_type *self_dot0, const kernel_type *self_dot1, kernel_type *dot_product, int m, int n,
                      kernel_type gamma) {
        //m rows of kernel matrix, where m is the working set size; n is the number of training instances
        KERNEL_LOOP(idx, m * n) {
            int i = idx / n;//i is row id
            int j = idx % n;//j is column id
            dot_product[idx] = expf(-(self_dot0[i] + self_dot1[j] - dot_product[idx] * 2) * gamma);
        }
    }

    __global__ void
    kernel_RBF_kernel(const int *self_dot0_idx, const kernel_type *self_dot1, kernel_type *dot_product, int m, int n,
                      kernel_type gamma) {
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
                             const kernel_type *k_mat, float_type *dec_values, int n_classes, int n_instances) {
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
                    const kernel_type *k_values = &k_mat[idx * total_sv];
                    double sum = 0;
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
    kernel_poly_kernel(kernel_type *dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        KERNEL_LOOP(idx, mn) {
            dot_product[idx] = powf(gamma * dot_product[idx] + coef0, degree);
        }
    }

    __global__ void kernel_sigmoid_kernel(kernel_type *dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        KERNEL_LOOP(idx, mn) {
            dot_product[idx] = tanhf(gamma * dot_product[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances) {
        SAFE_KERNEL_LAUNCH(kernel_sum_kernel_values, coef.device_data(), total_sv, sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), k_mat.device_data(), dec_values.device_data(),
                           n_classes, n_instances);

    }

    void
    get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                        const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n) {
        SAFE_KERNEL_LAUNCH(kernel_get_working_set_ins, val.device_data(), col_ind.device_data(), row_ptr.device_data(),
                           data_row_idx.device_data(), data_rows.device_data(), m, n);

    }

    void
    RBF_kernel(const SyncArray<kernel_type> &self_dot0, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n,
               kernel_type gamma) {
        SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, self_dot0.device_data(), self_dot1.device_data(),
                           dot_product.device_data(), m, n, gamma);
    }

    void
    RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, self_dot0_idx.device_data(), self_dot1.device_data(),
                           dot_product.device_data(), m, n, gamma);
    }

    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        SAFE_KERNEL_LAUNCH(kernel_poly_kernel, dot_product.device_data(), gamma, coef0, degree, mn);
    }

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        SAFE_KERNEL_LAUNCH(kernel_sigmoid_kernel, dot_product.device_data(), gamma, coef0, mn);
    }

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    bool cusparse_init;
    //m for instance; n for get_rows num; k for feature num; nnz for number of nonzero
    void dns_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result) {
        if (!cusparse_init) {
            cusparseCreate(&handle);
            cusparseCreateMatDescr(&descr);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparse_init = true;
        }
        kernel_type one(1);
        kernel_type zero(0);

#if (CUDART_VERSION >= 11000)

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
        
#ifdef USE_DOUBLE
        cudaDataType data_type = CUDA_R_64F;
#else//kernel type is float
        cudaDataType data_type = CUDA_R_32F;
#endif

        // cusparseCreateCsr(&matA, m, k, nnz, (void*)csr_row_ptr.device_data(), (void*)csr_col_ind.device_data(),
        //                   (void*)csr_val.device_data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        //                   CUSPARSE_INDEX_BASE_ZERO, data_type);
        // cusparseCreateDnMat(&matB, n, k, n, (void*)dense_mat.device_data(), data_type, CUSPARSE_ORDER_COL);
        // cusparseCreateDnMat(&matC, m, n, m, (void*)result.device_data(), data_type, CUSPARSE_ORDER_COL);
        

        
        // size_t buffer_size = 0;
        // cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        //                        &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1,
        //                        &buffer_size);

        // void *p_buffer = nullptr;
        // g_allocator.DeviceAllocate(&p_buffer, buffer_size);
        
        // cudaMalloc((void**)&p_buffer, buffer_size);
        
        // cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        //            &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1, p_buffer);
        // cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        //            &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1, p_buffer);
        
        // cudaFree(p_buffer);
        // g_allocator.DeviceFree(p_buffer);
        // cusparseDestroySpMat(matA);
        // cusparseDestroyDnMat(matB);
        // cusparseDestroyDnMat(matC);


        //test ror major

        //store tmp result and tmp trans dense_mat
        // cusparseDnMatDescr_t matB2,matC2;
        // // LOG(INFO)<<"m is "<<m<<" n is "<<n;
        // SyncArray<kernel_type> tmp_result(m*n);
        // SyncArray<kernel_type> tmp_dense1(n*k);
        // SyncArray<kernel_type> tmp_dense2(k*n);

        // tmp_dense2.copy_from(dense_mat);
        
        // kernel_type* h_tmp1 = tmp_dense1.host_data();
        // kernel_type* h_tmp2 = tmp_dense2.host_data();

        // for(int i=0;i<n;i++){
        //     for(int j=0;j<k;j++){
        //         h_tmp1[i*k+j] = h_tmp2[j*n+i];
        //     }

        // }
        // cusparseCreateDnMat(&matB2, n, k, k, (void*)tmp_dense1.device_data(), data_type, CUSPARSE_ORDER_ROW);//CUSPARSE_ORDER_ROW
        // cusparseCreateDnMat(&matC2, m, n, n, (void*)tmp_result.device_data(), data_type, CUSPARSE_ORDER_ROW);
        // //test row major
        // size_t buffer_size2 = 0;
        // cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        //                         &one, matA, matB2, &zero, matC2, data_type, CUSPARSE_SPMM_CSR_ALG2,
        //                         &buffer_size2);

        // void *p_buffer2 = nullptr;
        // g_allocator.DeviceAllocate(&p_buffer2, buffer_size2);
        
        // cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        //             &one, matA, matB2, &zero, matC2, data_type, CUSPARSE_SPMM_CSR_ALG2, p_buffer2);


        // g_allocator.DeviceFree(p_buffer2);
        // cusparseDestroySpMat(matA);
        // cusparseDestroyDnMat(matB2);
        // cusparseDestroyDnMat(matC2);



        
        //check result
        // kernel_type* h_res = result.host_data();
        // kernel_type* h_tmp = tmp_result.host_data();
        // kernel_type diff = 0;
        // for(int i = 0;i<n;i++){
        //     for(int j= 0;j<m;j++){
        //         h_res[i*m+j] = h_tmp[j*n+i];
        //         //diff+=fabs(h_res[i*m+j]-h_tmp[j*n+i]);
        //         //LOG(INFO)<<"result is "<<h_res[i*m+j]<<" tmp res is "<<h_tmp[j*n+i];
        //     }
        // }
        // //LOG(INFO)<<"diff is "<<diff;

        // result.device_data();
            
        
        //try cub
        
        //def
       SyncArray<kernel_type> d_values_;
       SyncArray<int> d_column_indices_;//index of each value of all the instances
       SyncArray<int> d_row_offsets_;//the start positions of the instances
       SyncArray<kernel_type> d_mat_x_;
       SyncArray<kernel_type> d_mat_xt;
       //kernel_type *d_vector_y = result.device_data();

       //resize
       d_values_.resize(csr_val.size());
       d_column_indices_.resize(csr_col_ind.size());
       d_row_offsets_.resize(csr_row_ptr.size());
       d_mat_x_.resize(dense_mat.size());
       d_mat_xt.resize(dense_mat.size());
       //copy
       d_values_.copy_from(csr_val );
       d_column_indices_.copy_from(csr_col_ind);
       d_row_offsets_.copy_from(csr_row_ptr);
       d_mat_x_.copy_from(dense_mat);
       d_mat_xt.copy_from(dense_mat);


       kernel_type *d_values = d_values_.device_data();
       int *d_row_offsets = d_row_offsets_.device_data();
       int *d_column_indices = d_column_indices_.device_data();

       kernel_type* h_mat_x = d_mat_x_.host_data();
       kernel_type* h_mat_xt = d_mat_xt.host_data();

       //trans

       TDEF(trans)
       TSTART(trans)
       for(int i=0;i<n;i++){
           for(int j=0;j<k;j++){
               h_mat_x[i*k+j] = h_mat_xt[j*n+i];
           }

       }
       TEND(trans)
       time1+=TINT(trans);
       //for(int i = 0;i<5;i++){
       //    LOG(INFO)<<dense_mat.host_data()[i*n+1]<<" "<<h_mat_x[i+1*k];
       //}
       //LOG(INFO)<<"done";


       kernel_type *d_mat_x = d_mat_x_.device_data();
       
       // kernel_type *d_vector_y;
       //cudaMalloc((void**)&d_vector_y,m*n*sizeof(kernel_type));
       // cudaMallocManaged((void**)&d_vector_y,m*n*sizeof(kernel_type));
           
       SyncArray<kernel_type> tmp_res(m*n);
       kernel_type *d_vector_y = tmp_res.device_data();
       //kernel_type *h_tmp = tmp_res.host_data();
       void* d_temp_storage = NULL;
       size_t temp_storage_bytes = 0;
       cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
                               d_row_offsets, d_column_indices, d_mat_x, d_vector_y,
                               m, k, nnz);

       g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
       
       // cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
       //                         d_row_offsets, d_column_indices, d_mat_x, d_vector_y,
       //                         m, k, nnz);
       


        for(int i=0;i<n;i++){

           cub::DeviceSpmv::CsrMV( d_temp_storage, temp_storage_bytes, d_values,
                                   d_row_offsets, d_column_indices, d_mat_x+i*k, d_vector_y+i*m,
                                   m, k, nnz);
           
        }
        
        cudaDeviceSynchronize();
        //copy 
        result.copy_from(tmp_res);
        //check
        // kernel_type *h_res = result.host_data();
        // kernel_type *h_tmp = tmp_res.host_data();
        // //
        // float r = 0;
        // for(int i = 0;i<n*k;i++){
        //    // r+=fabs(h_tmp[i]);
        //    r +=fabs(h_res[i]-h_tmp[i]);
        // }
        // LOG(INFO)<<"n is "<<n<<" k is "<<k<<" res sum is "<<r;
        
        g_allocator.DeviceFree(d_temp_storage);
        


        
        
#else

#ifdef USE_DOUBLE
        cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                        m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                        csr_col_ind.device_data(),
                        dense_mat.device_data(), n, &zero, result.device_data(), m);
#else//kernel type is float
        cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                        m, n, k, nnz, &one, descr, csr_val.device_data(), csr_row_ptr.device_data(),
                        csr_col_ind.device_data(),
                        dense_mat.device_data(), n, &zero, result.device_data(), m);

        //cusparseScsrmm return row-major matrix, so no transpose is needed
#endif // ifdef USE_DOUBLE

#endif // if CUDART_VERSION >= 11000
    }
}
