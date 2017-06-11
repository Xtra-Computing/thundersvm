//@brief:kernel functions to help complete gpu_sigmoid_train();
//@created on 16/12/6
//@author:yawen Chen

#ifndef SIGMOIDTRAINGPUHELPER_H_
#define SIGMOIDTRAINGPUHELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
//#include "svm-shared/host_constant.h"
#include "../svm-shared/host_constant.h"


__global__ void dev_getfApB(int l, real *dev_fApB,const real *dev_dec_values ,real A,real B);
__global__ void dev_getnewfApB(int l, real *dev_fApB,const real *dev_dec_values ,real *A,real *B);
__global__ void  dev_getfApB_fval(real *dev_fval, real *dev_labels, real *dev_t, real *dev_dec_values,real *dev_fApB, real A,real B, real hiTarget,real loTarget, int l);
__global__ void dev_get_fval_sum(real *dev_fval);
__global__ void dev_getpq(int l,real *dev_t,real *dev_fApB,real *dev_p,real *dev_q,real *dev_d1,real *dev_d2,real *dev_h11,real *dev_h21,real *dev_g1,real *dev_dec_values);
__global__ void dev_paral_red_sum(real *dev_arr,real *dev_sum,int l);
__global__ void dev_get_sum(real *dev_sum,real *dev_arr,int blocknum);
__global__ void dev_get_det(real sigma, real *dev_h11,real *dev_d2, real *dev_h21,real *dev_det);
__global__ void dev_getdA(real *dev_dA,real *dev_det,real *dev_h22, real *dev_h21,real *dev_g1,real *dev_g2);
__global__ void dev_getdB(real *dev_dB,real *dev_det,real *dev_h11, real *dev_h21,real *dev_g1,real *dev_g2);
__global__ void dev_getgd(real *dev_gd,real *dev_dA,real *dev_dB,real *dev_g1,real *dev_g2);
__global__ void dev_getnewf(int l, real *dev_fApB,real *dev_t,real *dev_newf);
__global__ void dev_updateAB(real *dev_newA,real *dev_newB,real A,real B,real stepsize,real *dev_dA,real *dev_dB);
__global__ void dev_getprior(real *dev_labels,int l,real *dev_prior1,real *dev_prior0)
#endif