//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/syncarray.h>
#include <cusparse.h>
#include "thundersvm/kernel/kernelmatrix_kernel.h"
#include <thundersvm/config.h>

#include <cublas_v2.h>
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

        }
    }


    __global__ void
    kernel_get_working_set_ins_dns(const kernel_type *val, const int *data_row_idx,
                               kernel_type *data_rows,
                               int m, int n,int n_instances) {
        
        KERNEL_LOOP(i, m) {
            int row = data_row_idx[i];
            for (int j = 0; j < n; ++j) {

                //data_rows[i*n + j] = val[row*n+j]; // row-major for cublas

                // data_rows[i + j*m] = val[row*n+j]; // col-major for cublas

                data_rows[i + j*m] = val[row+j*n_instances]; // col-major for cublas, val col major

            }
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
    get_working_set_ins_dns(const SyncArray<kernel_type> &val, 
                            const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n,int n_instances){

        SAFE_KERNEL_LAUNCH(kernel_get_working_set_ins_dns, val.device_data(),
                           data_row_idx.device_data(), data_rows.device_data(), m, n,n_instances);
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

        cusparseCreateCsr(&matA, m, k, nnz, (void*)csr_row_ptr.device_data(), (void*)csr_col_ind.device_data(),
                          (void*)csr_val.device_data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, data_type);
        cusparseCreateDnMat(&matB, n, k, n, (void*)dense_mat.device_data(), data_type, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&matC, m, n, m, (void*)result.device_data(), data_type, CUSPARSE_ORDER_COL);
        
       
        
        size_t buffer_size = 0;
        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                               &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1,
                               &buffer_size);

        void *p_buffer = nullptr;
        
        cudaMalloc((void**)&p_buffer, buffer_size);
        
        cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                   &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1, p_buffer);
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                   &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1, p_buffer);
        
        cudaFree(p_buffer);
        
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);

        cudaDeviceSynchronize();    

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



    //dns dns mul

    cublasHandle_t handle_blas;
    bool cublas_init;
    void dns_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_a,const SyncArray<kernel_type> &dense_b,kernel_type beta, 
                     SyncArray<kernel_type> &result){

        if (!cublas_init) {
            cublasCreate(&handle_blas);
            cublas_init = true;
        }

        kernel_type alpha=1.0;
        const kernel_type* d_dense_a = dense_a.device_data();
        const kernel_type* d_dense_b = dense_b.device_data();

        // cublasSgemm(handle_blas,CUBLAS_OP_T,CUBLAS_OP_N, m, n, k,&alpha,dense_a.device_data(), k, dense_b.device_data(), k,&beta, result.device_data(), m);

        //dense b :k*n
        // cublasSgemm(handle_blas,CUBLAS_OP_T,CUBLAS_OP_T, m, n, k,&alpha,dense_a.device_data(), k, dense_b.device_data(), n,&beta, result.device_data(), m);
        cublasSgemm(handle_blas,CUBLAS_OP_N,CUBLAS_OP_T, m, n, k,&alpha,dense_a.device_data(), m, dense_b.device_data(), n,&beta, result.device_data(), m);
        

    }

    //csr csr mul
    void csr_csr_mul_cuda(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result){

        if (!cusparse_init) {
            cusparseCreate(&handle);
            cusparseCreateMatDescr(&descr);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparse_init = true;
        }

        
        kernel_type alpha(1);
        kernel_type beta(0);
        cudaDataType data_type = CUDA_R_32F;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

        cusparseSpMatDescr_t matA, matB, matC;
        
        cusparseDnMatDescr_t tmp_mat,result_mat;

        

        void* dBuffer = NULL;
        size_t bufferSize = 0;

        void*  dBuffer1 = NULL, *dBuffer2 = NULL,*dBuffer3 = NULL;
        size_t bufferSize1 = 0, bufferSize2 = 0 ,bufferSize3= 0;

        

        int *tmp_csr_row;
        cudaMalloc((void**) &tmp_csr_row,(k + 1) * sizeof(int));

        //create matrix
        cusparseCreateCsr(&matA, m, k, nnz, (void*)csr_row_ptr.device_data(), (void*)csr_col_ind.device_data(),
                          (void*)csr_val.device_data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, data_type);

        cusparseCreateCsr(&matB, k, n, 0,
                                      tmp_csr_row, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, data_type);

        cusparseCreateCsr(&matC, m, n, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, data_type);

        cusparseCreateDnMat(&result_mat, m, n, m, (void*)result.device_data(),
                                        data_type, CUSPARSE_ORDER_COL);

        
     
        //dense转化为csr格式, shape k*n
        
        cusparseCreateDnMat(&tmp_mat, k, n, n, (void*)dense_mat.device_data(), data_type, CUSPARSE_ORDER_ROW);

        cusparseDenseToSparse_bufferSize(
                                        handle, tmp_mat, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize);
        cudaMalloc((void**)&dBuffer, bufferSize);


        cusparseDenseToSparse_analysis(handle, tmp_mat, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer);

        int64_t num_rows_tmp, num_cols_tmp, nnz_tmp;
        int *d_csr_columns;
        
        kernel_type* d_csr_values;

        cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,&nnz_tmp);
        
        cudaMalloc((void**) &d_csr_columns, nnz_tmp * sizeof(int));
        cudaMalloc((void**) &d_csr_values,  nnz_tmp * sizeof(kernel_type));
        

        cusparseCsrSetPointers(matB, tmp_csr_row, d_csr_columns,d_csr_values);
        cusparseDenseToSparse_convert(handle, tmp_mat, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer);


        //timing
        cudaEvent_t start_event, stop_event;
        float cuda_elapsed_ms  = 0;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, NULL);
        //csr csr mul

        cusparseSpGEMMDescr_t spgemmDesc;
        cusparseSpGEMM_createDescr(&spgemmDesc);

        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      data_type, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL);
        cudaMalloc((void**) &dBuffer1, bufferSize1);

        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      data_type, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1);

        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               data_type, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL);

        cudaMalloc((void**) &dBuffer2, bufferSize2);

        cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           data_type, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2);

        int64_t C_num_rows1, C_num_cols1, C_nnz1;
        cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,&C_nnz1);

        int *dC_csrOffsets,*dC_columns;
        kernel_type* dC_values;

        cudaMalloc((void**) &dC_csrOffsets, (m+1) * sizeof(int));
        cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int));
        cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(kernel_type));
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values);
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            data_type, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

        cudaEventRecord(stop_event, NULL);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&cuda_elapsed_ms, start_event,stop_event);
        LOG(INFO)<<"csr csr mul time is "<<cuda_elapsed_ms;
        //csr to dns

        cusparseSparseToDense_bufferSize(
                                        handle, matC, result_mat,
                                        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                        &bufferSize3);
        cudaMalloc(&dBuffer3, bufferSize3);
        
        cusparseSparseToDense(handle, matC, result_mat,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer3);
        
        

        cusparseDestroySpMat(matA);
        cusparseDestroySpMat(matB);
        cusparseDestroySpMat(matC);
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cusparseDestroyDnMat(result_mat);
        cudaFree(dBuffer);
        cudaFree(dBuffer1);
        cudaFree(dBuffer2);
        cudaFree(dBuffer3);
        cudaFree(tmp_csr_row);
        cudaFree(d_csr_columns);
        cudaFree(d_csr_values);
        cusparseDestroyDnMat(tmp_mat);
        cudaFree(dBuffer);
        cudaFree(dC_csrOffsets);
        cudaFree(dC_columns);
        cudaFree(dC_values);




    } 


    //bsr dns mul

    void bsr_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &bsr_val,
                     const SyncArray<int> &bsr_row_ptr, const SyncArray<int> &bsr_col_ind, 
                     SyncArray<kernel_type> &result) {
        if (!cusparse_init) {
            cusparseCreate(&handle);
            cusparseCreateMatDescr(&descr);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparse_init = true;
        }
        kernel_type alpha(1);
        kernel_type beta(0);

        cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

        int nnzb = bsr_col_ind.size();
        int mb = bsr_row_ptr.size()-1;
        int blockSize = sqrt(bsr_val.size()/nnzb);
        int nb = (k+blockSize-1)/blockSize;

        //mul
        cusparseSbsrmm(handle,
               dir,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_TRANSPOSE,
               mb, n, nb, nnzb, &alpha,
               descr, bsr_val.device_data(), bsr_row_ptr.device_data(), bsr_col_ind.device_data(), blockSize,
               dense_mat.device_data(), n,
               &beta, result.device_data(), m);
    }



    void csc_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csc_val,
                     const SyncArray<int> &csc_row_ptr, const SyncArray<int> &csc_col_ind, int nnz,
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

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;

        cudaDataType data_type = CUDA_R_32F;

        // cusparseCreateCsr(&matA, k, m, nnz, (void*)csc_col_ind.device_data(), (void*)csc_row_ptr.device_data(),
        //                   (void*)csc_val.device_data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        //                   CUSPARSE_INDEX_BASE_ZERO, data_type);

        cusparseCreateCsc(&matA, m, k, nnz, (void*)csc_col_ind.device_data(), (void*)csc_row_ptr.device_data(),
                          (void*)csc_val.device_data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, data_type);
        cusparseCreateDnMat(&matB, n, k, n, (void*)dense_mat.device_data(), data_type, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&matC, m, n, m, (void*)result.device_data(), data_type, CUSPARSE_ORDER_COL);

        size_t buffer_size = 0;
        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1,
                                &buffer_size);

        void *p_buffer = nullptr;
        cudaMalloc((void**)&p_buffer, buffer_size);

        cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                   &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1, p_buffer);
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                    &one, matA, matB, &zero, matC, data_type, CUSPARSE_SPMM_CSR_ALG1, p_buffer);

        cudaFree(p_buffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
    }

}


