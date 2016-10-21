/**
 * kernelCalGPUHelper.h
 * Created on: May 29, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef KERNELCALGPUHELPER_H_
#define KERNELCALGPUHELPER_H_

#include <cuda_runtime.h>
#include <cuda.h>

#include <math.h>
#include "../gpu_global_utility.h"

/******************* Guassian Kernel *******************************/
/*
 * @brief: compute one Hessian row
 * @param: pfDevSamples: data of samples. One dimension array represents a matrix
 * @param: pfDevTransSamples: transpose data of samples
 * @param: pfDevHessianRows: a Hessian row. the final result of this function
 * @param: nNumofSamples: the number of samples
 * @param: nNumofDim: the number of dimensions for samples
 * @param: nStartRow: the Hessian row to be computed
 */
__device__ void RBFOneRow(float_point *pfDevSamples, float_point *pfDevTransSamples,
						  float_point *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, float_point fGamma);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void RBFKernel(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nNumofRows, int nStartRow, float_point fGamma);

__global__ void ObtainRBFKernel(float_point *pfDevHessianRows,float_point *selfDot, int nNumofSamples,
								int nNumofRows, float_point fGamma, int nStartRow, int nStartCol);
__global__ void UpdateDiag(float_point *pfDevHessianRows, int nNumofSamples, int nNumofRows);

/****************** Linear Kernel ********************************/
/*
 * @brief: compute one Hessian row
 * @param: pfDevSamples: data of samples. One dimension array represents a matrix
 * @param: pfDevTransSamples: transpose data of samples
 * @param: pfDevHessianRows: a Hessian row. the final result of this function
 * @param: nNumofSamples: the number of samples
 * @param: nNumofDim: the number of dimensions for samples
 * @param: nStartRow: the Hessian row to be computed
 */
__device__ void LinearOneRow(float_point *pfDevSamples, float_point *pfDevTransSamples,
						  float_point *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void LinearKernel(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nStartRow);

/***************** Polynomial Kernel ******************************/
/*
 * @brief: compute one Hessian row
 * @param: pfDevSamples: data of samples. One dimension array represents a matrix
 * @param: pfDevTransSamples: transpose data of samples
 * @param: pfDevHessianRows: a Hessian row. the final result of this function
 * @param: nNumofSamples: the number of samples
 * @param: nNumofDim: the number of dimensions for samples
 * @param: nStartRow: the Hessian row to be computed
 */
__device__ void PolynomialOneRow(float_point *pfDevSamples, float_point *pfDevTransSamples,
						  float_point *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, float_point fR, float_point fDegreee);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void PolynomialKernel(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nStartRow, float_point fDegree);

/********************* Sigmoid Kernel ******************************/
/*
 * @brief: compute one Hessian row
 * @param: pfDevSamples: data of samples. One dimension array represents a matrix
 * @param: pfDevTransSamples: transpose data of samples
 * @param: pfDevHessianRows: a Hessian row. the final result of this function
 * @param: nNumofSamples: the number of samples
 * @param: nNumofDim: the number of dimensions for samples
 * @param: nStartRow: the Hessian row to be computed
 */
__device__ void SigmoidOneRow(float_point *pfDevSamples, float_point *pfDevTransSamples,
						  float_point *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, float_point fR, float_point fDegreee);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void SigmoidKernel(float_point *pfDevSamples, float_point *pfDevTransSamples, float_point *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nStartRow, float_point fDegree);

#endif /* KERNELCALGPUHELPER_H_ */
