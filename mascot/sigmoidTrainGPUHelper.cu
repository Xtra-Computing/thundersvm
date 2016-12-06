

//#include "svm.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include "../svm-shared/host_constant.h"
//#include "svm-shared/host_constant.h"

__global__ void dev_getfApB(int l, float *dev_fApB,const float *dev_dec_values ,float A,float B){
	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<l)
		dev_fApB[idx] = dev_dec_values[idx]*A+B;
}

__global__ void dev_getnewfApB(int l, float *dev_fApB,const float *dev_dec_values ,float *A,float *B){
	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<l)
		dev_fApB[idx] = dev_dec_values[idx]*A[0]+B[0];
}
__global__ void  dev_getfApB_fval(float *dev_fval, float *dev_labels, float *dev_t, float *dev_dec_values,float *dev_fApB, float A,float B, float hiTarget,float loTarget, int l){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;
	//extern __shared__ float sharefval[];
	if(idx<l){
		if (dev_labels[idx]>0)
			dev_t[idx]=hiTarget;
		else 
			dev_t[idx]=loTarget;
		dev_fApB[idx] = dev_dec_values[idx]*A+B;
		if (dev_fApB[idx]>=0)
			//sharefval[tid] = dev_t[idx]*dev_fApB[idx] + log(1+exp(-dev_fApB[idx]));
			dev_fval[idx]	= dev_t[idx]*dev_fApB[idx] + log(1+exp(-dev_fApB[idx]));
		else
			//sharefval[tid] = (dev_t[idx] - 1)*dev_fApB[idx] +log(1+exp(dev_fApB[idx]));
			dev_fval[idx]	= (dev_t[idx] - 1)*dev_fApB[idx] +log(1+exp(dev_fApB[idx]));
	
	}
}

__global__ void dev_get_fval_sum(float *dev_fval){
	for(int i=1;i<blockDim.x;i++)
		dev_fval[0]+=dev_fval[i];
}

__global__ void dev_getpq(int l,float *dev_t,float *dev_fApB,float *dev_p,float *dev_q,float *dev_d1,float *dev_d2,float *dev_h11,float *dev_h21,float *dev_g1,float *dev_dec_values){
	int idx=threadIdx.x+blockDim.x*blockIdx.x;
	if(idx<l){
		if (dev_fApB[idx] >= 0)
			{
				dev_p[idx]=exp(-dev_fApB[idx])/(1.0+exp(-dev_fApB[idx]));
				dev_q[idx]=1.0/(1.0+exp(-dev_fApB[idx]));
			}
		else
			{
				dev_p[idx]=1.0/(1.0+exp(dev_fApB[idx]));
				dev_q[idx]=exp(dev_fApB[idx])/(1.0+exp(dev_fApB[idx]));
			}
		dev_d2[idx]=dev_p[idx]*dev_q[idx];
		dev_h11[idx]=dev_dec_values[idx]*dev_dec_values[idx]*dev_d2[idx];
		dev_h21[idx]=dev_dec_values[idx]*dev_d2[idx];
		dev_d1[idx]=dev_t[idx]-dev_p[idx];
		dev_g1[idx]=dev_dec_values[idx]*dev_d1[idx];
		//dev_g2[idx]=dev_d1[idx];
	}
}

__global__ void dev_paral_red_sum(float *dev_arr,float *dev_sum,int l){
	extern __shared__ float s_arr[];
	int tid=threadIdx.x;
	int blockSize=blockDim.x;
	int i=(blockSize*2)*blockIdx.x+tid;
	int gridSize=blockSize*2*gridDim.x;
	s_arr[tid]=0;
	while(i<l){
		s_arr[tid]+=dev_arr[i]+dev_arr[i+blockSize];
		i+=gridSize;
	}
	__syncthreads();

	if( blockSize>=512){
		if(tid<256){
			s_arr[tid]+=s_arr[tid+256];
		}
		__syncthreads();
	}
	if( blockSize>=256){
		if(tid<128){
			s_arr[tid]+=s_arr[tid+128];
		}
		__syncthreads();
	}
	if( blockSize>=128){
		if(tid<64){
			s_arr[tid]+=s_arr[tid+64];
		}
		__syncthreads();
	}

	if(tid<32){
		if(blockSize>=64){
			s_arr[tid]+=s_arr[tid+32];
			__syncthreads();}
		if(blockSize>=32){
			s_arr[tid]+=s_arr[tid+16];
			__syncthreads();}
		if(blockSize>=16){
			s_arr[tid]+=s_arr[tid+8];
			__syncthreads();}
		if(blockSize>=8){
			s_arr[tid]+=s_arr[tid+4];
			__syncthreads();}
		if(blockSize>=4){
			s_arr[tid]+=s_arr[tid+2];
		__syncthreads();}
		if(blockSize>=2){
			s_arr[tid]+=s_arr[tid+1];
			__syncthreads();}
	}
	
	
	if(tid==0) 
		dev_sum[blockIdx.x]=s_arr[0];
}

__global__ void dev_get_sum(float *dev_sum,float *dev_arr,int blocknum){
	for(int i=1;i<blocknum;i++)
		dev_sum[0]+=dev_sum[i];
	dev_arr[0]=dev_sum[0];
}

__global__ void dev_get_det(float sigma, float *dev_h11,float *dev_d2, float *dev_h21,float *dev_det){
	dev_h11[0]+=sigma;
	dev_d2[0]+=sigma;
	dev_det[0]=dev_h11[0]*dev_d2[0]-dev_h21[0]*dev_h21[0];
}

__global__ void dev_getdA(float *dev_dA,float *dev_det,float *dev_h22, float *dev_h21,float *dev_g1,float *dev_g2){
	dev_dA[0]=-(dev_h22[0]*dev_g1[0] - dev_h21[0] * dev_g2[0]) / dev_det[0];
	
}
__global__ void dev_getdB(float *dev_dB,float *dev_det,float *dev_h11, float *dev_h21,float *dev_g1,float *dev_g2){
	dev_dB[0]=-(-dev_h21[0]*dev_g1[0]+ dev_h11[0] * dev_g2[0]) / dev_det[0];
}
__global__ void dev_getgd(float *dev_gd,float *dev_dA,float *dev_dB,float *dev_g1,float *dev_g2){
	dev_gd[0]=dev_g1[0]*dev_dA[0]+dev_g2[0]*dev_dB[0];
}


__global__ void dev_getnewf(int l, float *dev_fApB,float *dev_t,float *dev_newf){
	int idx=blockDim.x*blockIdx.x+threadIdx.x;
	if(idx<l){
		if (dev_fApB[idx] >= 0)
			dev_newf[idx] = dev_t[idx]*dev_fApB[idx] + log(1+exp(-dev_fApB[idx]));
		else
			dev_newf[idx] = (dev_t[idx] - 1)*dev_fApB[idx] +log(1+exp(dev_fApB[idx]));
	}
}

__global__ void dev_updateAB(float *dev_newA,float *dev_newB,float A,float B,float stepsize,float *dev_dA,float *dev_dB){
	switch(threadIdx.x){
		case 0:{
			dev_newA[0] = A + stepsize * dev_dA[0];
			break;
		}
		case 1:{
			dev_newB[0] = B + stepsize * dev_dB[0];
			break;
		}
	}
	
}
