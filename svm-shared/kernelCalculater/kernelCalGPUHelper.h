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
__device__ void RBFOneRow(real *pfDevSamples, real *pfDevTransSamples,
						  real *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, real fGamma);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void RBFKernel(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nNumofRows, int nStartRow, real fGamma);

__global__ void ObtainRBFKernel(real *pfDevHessianRows,real *selfDot, int nNumofSamples,
								int nNumofRows, real fGamma, int nStartRow, int nStartCol);
__global__ void UpdateDiag(real *pfDevHessianRows, int nNumofSamples, int nNumofRows);

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
__device__ void LinearOneRow(real *pfDevSamples, real *pfDevTransSamples,
						  real *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void LinearKernel(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
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
__device__ void PolynomialOneRow(real *pfDevSamples, real *pfDevTransSamples,
						  real *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, real fR, real fDegreee);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void PolynomialKernel(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nStartRow, real fDegree);

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
__device__ void SigmoidOneRow(real *pfDevSamples, real *pfDevTransSamples,
						  real *pfDevHessianRows, int nNumofSamples, int nNumofDim,
						  int nStartRow, real fR, real fDegreee);


//a few blocks compute one row of the Hessian matrix. The # of threads invovled in a row is equal to the # of samples
//one thread an element of the row
//the # of thread is equal to the # of dimensions or the available size of shared memory
__global__ void SigmoidKernel(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
						  int nNumofSamples, int nNumofDim, int nStartRow, real fDegree);

#endif /* KERNELCALGPUHELPER_H_ */
