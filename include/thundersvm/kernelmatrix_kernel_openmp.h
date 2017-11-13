#include <omp.h>
#include <iostream>
#include "thundersvm/thundersvm.h"
#include "thundersvm/clion_cuda.h"
//typedef float real;
void kernel_get_working_set_ins_openmp(const real *val, const int *col_ind, const int *row_ptr, const int *data_row_idx,
                           real *data_rows,
                           int m);
void kernel_sum_kernel_values_openmp(const real *k_mat, int n_instances, int n_sv_unique, int n_bin_models,
                                         const int *sv_index, const real *coef, const int *sv_start,
                                         const int *sv_count,
                                         const real *rho, real *dec_values);
void kernel_RBF_kernel_openmp(const real *self_dot0, const real *self_dot1, real *dot_product, int m, int n, real gamma);
void kernel_RBF_kernel_openmp(const int *self_dot0_idx, const real *self_dot1, real *dot_product, int m, int n, real gamma);
void kernel_sum_kernel_values_openmp(const real *k_mat, int n_instances, int n_sv_unique, int n_bin_models,
                                         const int *sv_index, const real *coef, const int *sv_start,
                                         const int *sv_count,
                                         const real *rho, real *dec_values);
void kernel_poly_kernel_openmp(real *dot_product, real gamma, real coef0, int degree, int mn);
void kernel_sigmoid_kernel_openmp(real *dot_product, real gamma, real coef0, int mn);