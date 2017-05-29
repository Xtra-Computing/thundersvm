/**
 * kernelCalculater.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 * @brief: this file contains classes for computing different kind of kernels
 **/

#ifndef KERNELCALCULATER_H_
#define KERNELCALCULATER_H_

#include "../gpu_global_utility.h"
#include "kernelCalGPUHelper.h"
#include "../constant.h"
#include "../../SharedUtility/CudaMacro.h"

#include <iostream>
using namespace std;
using std::cerr;
using std::cout;
using std::string;

/*
 * abstract class for Kernel calculation
 */
class CKernelCalculater
{
public:
	CKernelCalculater(){}
	virtual ~CKernelCalculater(){}

	virtual bool GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, real *pfHessianDiag) = 0;

	virtual bool ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows, real *pfDevSelfDot,
									const int &nNumofSamples, const int &nNumofDim,
									const int &nNumofRows, int nStartRow, int nStartCol) = 0;

	virtual string GetType() = 0;

	void GetGPUSpec(dim3 &dimGrid, int &nBlockSize, const int &nNumofSamples, const int &nNumofRows)
	{
		/*compute the # of threads per block, the # of blocks per grid
		 * One grid for computing one Hessian Matrix row
		 */
		nBlockSize = ((nNumofSamples > BLOCK_SIZE) ? BLOCK_SIZE : nNumofSamples);
		int nNumofBlocks = Ceil(nNumofSamples, nBlockSize);
		int nGridDimX = 0, nGridDimY = 0, nGridDimZ = 0;

		//grid size in X dimension
		if(nNumofBlocks > NUM_OF_BLOCK)
			nGridDimX = NUM_OF_BLOCK;
		else
			nGridDimX = nNumofBlocks;
		//grid size in Y dimension (when X dimension is not enough to indicate blocks)
		nGridDimY = Ceil(nNumofBlocks, NUM_OF_BLOCK);
		//grid size in Z dimension, each element in Z dimension for one Hessian row
		if(nNumofRows > NUM_OF_BLOCK)
		{
			cerr << "the number of Hessian rows is too large" << endl;
		}
		else
		{
			nGridDimZ = nNumofRows;
		}

		dimGrid.x = nGridDimX;
		dimGrid.y = nGridDimY;
		dimGrid.z = nGridDimZ;

		//call kernel function, one thread computes one element of those Hessian Matrix rows
		if(cudaGetLastError() != cudaSuccess)
		{
			cerr << "cuda error before ComputeHessianRows" << endl;
		}
	}
};

/*
 * @brief: class for RBF (Guassian) Kernel
 */
class CRBFKernel: public CKernelCalculater
{
public:
	real m_fGamma;
public:
	CRBFKernel(real fGamma){m_fGamma = fGamma; cout << "gamma is: " << fGamma << endl;}
	~CRBFKernel(){}
	void SetGamma(real fGamma){m_fGamma = fGamma;}

	virtual string GetType(){return RBFKERNEL;}

	virtual bool GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, real *pfHessianDiag);

	virtual bool ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
									real *pfDevSelfDot, const int &nNumofCols, const int &nNumofDim,
									const int &nNumofRows, int nStartRow, int nStartCol);
	bool ComputeHessianMatrix(real *pfDevSamples, real *pfDevTransSamples, real *pfDevSelfDot,
							  real *pfDevHessianRows, const int &nNumofSamples,const int &nDim,
							  const int &nNumofRows, int nStartRow, int nStartCol);
	bool ComputeHessianRowsByCPU(real *pfSamples, real *pfHessianRows,
								 const int &nNumofSamples, const int &nNumofDim,
								 const int &nStartRow);
};

/*
 * @brief: class declaration for Linear Kernel
 */
class CLinearKernel: public CKernelCalculater
{
public:
	CLinearKernel(){}
	CLinearKernel(real){}
	~CLinearKernel(){}
	virtual string GetType(){return SVMLINEAR;}
	virtual bool GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, real *pfHessianDiag)
	{
		return true;
	}
	virtual bool ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
									const int &nNumofSamples, const int &nNumofDim,
									const int &nNumofRows, const int &nStartRow);
private:
	using CKernelCalculater::ComputeHessianRows;
};

/*
 * @brief: class declaration for Polynomial kernel
 */
class CPolynomialKernel: public CKernelCalculater
{
public:
	real m_r;
	real m_fDegree;
	CPolynomialKernel(){}
	CPolynomialKernel(real d){m_fDegree = d;}
	~CPolynomialKernel(){}
	virtual string GetType(){return SVMPOLYNOMIAL;}
	virtual bool GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, real *pfHessianDiag)
	{
		return true;
	}
	virtual bool ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
										const int &nNumofSamples, const int &nNumofDim,
										const int &nNumofRows, const int &nStartRow);
private:
	using CKernelCalculater::ComputeHessianRows;
};

/*
 * @brief: class declaration for Sigmoid kernel
 */
class CSigmoidKernel: public CKernelCalculater
{
public:
	real m_fCoef;
	CSigmoidKernel(){}
	CSigmoidKernel(real r){m_fCoef = r;}
	~CSigmoidKernel(){}
	virtual string GetType(){return SVMSIGMOID;}
	virtual bool GetHessianDiag(const string &strFileName, const int &nNumofTrainingSamples, real *pfHessianDiag)
	{
		return true;
	}
	virtual bool ComputeHessianRows(real *pfDevSamples, real *pfDevTransSamples, real *pfDevHessianRows,
											const int &nNumofSamples, const int &nNumofDim,
											const int &nNumofRows, const int &nStartRow);
private:
	using CKernelCalculater::ComputeHessianRows;
};


#endif /* KERNELCALCULATER_H_ */
