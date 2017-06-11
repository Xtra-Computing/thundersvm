/**
 * classificationKernel.h
 * Created on: Jun 6, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef CLASSIFICATIONKERNEL_H_
#define CLASSIFICATIONKERNEL_H_

//head files for GPU

#include "../svm-shared/gpu_global_utility.h"

/*
 * @brief: compute partial kernel sum for classification, as sometimes the # of SVs is very large
 * @param: pfSVYiAlphaHessian: an array storing yi*alpha*k(sv,x), where sv is a support vector. (in the format: T1 sv1 sv2...T2 sv1 sv2...)
 * @param: pfLocalPartialSum: a block level sum of yi*alpha*k(sv,x)
 * @param: nReduceStepSize: the step size for each pair of sum
 */
__global__ void ComputeKernelPartialSum(real* pfSVYiAlhpaHessian, int nNumofSVs,
										real* pfLocalPartialSum, int nReduceStepSize);

/*
 * @brief: compute global sum for each testing sample, and the final result
 * @param: pfClassificationResult: the result of the sum (the output of this function)
 * @param: fBias: the bias term of SVM
 * @param: pfPartialSum: the partial sum of block level
 */
__global__ void ComputeKernelGlobalSum(real *pfClassificationResult,
									   real fBias, real *pfPartialSum,
									   int nReduceStepSize);


/*
 * @brief: compute multiplication of two vectors
 * @output: result stores in pfVector1
 */
__global__ void VectorMul(real *pfVector1, int *pnVector2, int nNumofDim);

/*
 * @brief: compute a vector is multiplied by a matrix. This function does part of vector-matrix multiplication
 */
__global__ void VectorMatrixMul(real *pfVector, real *pfMatrix, int nNumofRow, int nNumofCol);

#endif /* CLASSIFICATIONKERNEL_H_ */
