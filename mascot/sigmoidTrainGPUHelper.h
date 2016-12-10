//@brief:kernel functions to help complete gpu_sigmoid_train();
//@created on 16/12/6
//@author:yawen Chen

#ifndef SIGMOIDTRAINGPUHELPER_H_
#define SIGMOIDTRAINGPUHELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
//#include "svm-shared/host_constant.h"
#include "../svm-shared/host_constant.h"


__global__ void dev_getfApB(int l, float_point *dev_fApB,const float_point *dev_dec_values ,float_point A,float_point B);
__global__ void dev_getnewfApB(int l, float_point *dev_fApB,const float_point *dev_dec_values ,float_point *A,float_point *B);
__global__ void  dev_getfApB_fval(float_point *dev_fval, float_point *dev_labels, float_point *dev_t, float_point *dev_dec_values,float_point *dev_fApB, float_point A,float_point B, float_point hiTarget,float_point loTarget, int l);
__global__ void dev_get_fval_sum(float_point *dev_fval);
__global__ void dev_getpq(int l,float_point *dev_t,float_point *dev_fApB,float_point *dev_p,float_point *dev_q,float_point *dev_d1,float_point *dev_d2,float_point *dev_h11,float_point *dev_h21,float_point *dev_g1,float_point *dev_dec_values);
__global__ void dev_paral_red_sum(float_point *dev_arr,float_point *dev_sum,int l);
__global__ void dev_get_sum(float_point *dev_sum,float_point *dev_arr,int blocknum);
__global__ void dev_get_det(float_point sigma, float_point *dev_h11,float_point *dev_d2, float_point *dev_h21,float_point *dev_det);
__global__ void dev_getdA(float_point *dev_dA,float_point *dev_det,float_point *dev_h22, float_point *dev_h21,float_point *dev_g1,float_point *dev_g2);
__global__ void dev_getdB(float_point *dev_dB,float_point *dev_det,float_point *dev_h11, float_point *dev_h21,float_point *dev_g1,float_point *dev_g2);
__global__ void dev_getgd(float_point *dev_gd,float_point *dev_dA,float_point *dev_dB,float_point *dev_g1,float_point *dev_g2);
__global__ void dev_getnewf(int l, float_point *dev_fApB,float_point *dev_t,float_point *dev_newf);
__global__ void dev_updateAB(float_point *dev_newA,float_point *dev_newB,float_point A,float_point B,float_point stepsize,float_point *dev_dA,float_point *dev_dB);
__global__ void dev_getprior(float_point *dev_labels,int l,float_point *dev_prior1,float_point *dev_prior0)
#endif