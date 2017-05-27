/*
 * subHessianCalculater.h
 *
 *  Created on: 10 Jan 2017
 *      Author: Zeyi Wen
 */

#ifndef SVM_SHARED_CACHE_SUBHESSIANCALCULATER_H_
#define SVM_SHARED_CACHE_SUBHESSIANCALCULATER_H_

#include <cublas_v2.h>
#include "../csrMatrix.h"
#include "../../mascot/svmProblem.h"
#include "../svmParam.h"

class SubHessianCalculator{
public:
    static void preComputeUniqueCache(int i, int j, const SvmProblem &subProblem,
    						   vector<real*> &devUniqueCache, vector<size_t> &sizeOfEachRowInUniqueCache,
							   vector<int> &numOfElementEachRowInUniqueCache, const SVMParam &param);
    static void preComputeSharedCache(vector<real*> &hostSharedCache, const SvmProblem &problem, const SVMParam &param);
	static void preComputeCache4BinaryProblem(real *devC, const SvmProblem &problem, const SVMParam &param);
    static void preComputeAndStoreInHost(real *hostHessianMatrix, const SvmProblem &problem, bool &preComputeInHost, const SVMParam &param);
    static void computeSubHessianMatrix(cusparseHandle_t handle, cusparseMatDescr_t descr,
    							 CSRMatrix &csrMatrix0, int n, CSRMatrix &csrMatrix1, int m, int k, real *devC,
                                 const SVMParam &param);

    static void prepareCSRContext(cusparseHandle_t &handle, cusparseMatDescr_t &descr);

    static void releaseCSRContext(cusparseHandle_t &handle, cusparseMatDescr_t &descr);
};

__global__ void RBFKernel(const real *selfDot0, const real *selfDot1,
						  real *dotProduct, int n, int m,
						  float gamma);

#endif /* SVM_SHARED_CACHE_SUBHESSIANCALCULATER_H_ */
