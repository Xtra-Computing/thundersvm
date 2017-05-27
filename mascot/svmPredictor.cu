/*
 * @brief: this file contains the definition of svm predictor class
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 */

#include "../svm-shared/gpu_global_utility.h"
#include "svmPredictor.h"
#include "../svm-shared/storageManager.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>

/*
 * @brief: read kernel values based on support vectors
 */
void CSVMPredictor::ReadKVbasedOnSV(real *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs, int nNumofTestSamples)
{
	FILE *pFile = fopen(HESSIAN_FILE, "rb");
	real *pfSVHessianSubRow = new real[nNumofTestSamples];
	real *pfHessianFullRow = new real[m_pHessianReader->m_nTotalNumofInstance];
	memset(pfSVHessianSubRow, 0, sizeof(real) * nNumofTestSamples);

	for(int i = 0; i < nNumofSVs; i++)
	{
		//read part of the Hessian Row

		if(m_pHessianReader->m_nNumofCachedHessianRow > pnSVSampleId[i])
		{
			//if the hessian row is in host memory
			long long nIndexofFirstElement;
			//only one if-statement holds, as testing samples are continuously allocated in RAM
			if(m_pHessianReader->m_nRowStartPos1 != -1)
			{
				nIndexofFirstElement = (long long)pnSVSampleId[i] * m_pHessianReader->m_nTotalNumofInstance +
			  									  m_pHessianReader->m_nRowStartPos1;
			}
			if(m_pHessianReader->m_nRowStartPos2 != -1)
			{
				nIndexofFirstElement = (long long)pnSVSampleId[i] * m_pHessianReader->m_nTotalNumofInstance +
												 m_pHessianReader->m_nRowStartPos2;
			}
			//copy the memory
			memcpy(pfSVHessianSubRow, m_pHessianReader->m_pfHessianRowsInHostMem + nIndexofFirstElement,
					nNumofTestSamples * sizeof(real));
		}
		else//the hessian row is in SSD
		{
			int nStartPos;
			if(m_pHessianReader->m_nRowStartPos1 != -1)
			{
				nStartPos = m_pHessianReader->m_nRowStartPos1;
			}
			else if(m_pHessianReader->m_nRowStartPos2 != -1)
			{
				nStartPos = m_pHessianReader->m_nRowStartPos2;
			}
			else
			{
				assert(0);
			}
			m_pHessianReader->ReadRow(pnSVSampleId[i], pfSVHessianSubRow);
			//int nIndexInSSD = pnSVSampleId[i] - m_pHessianOps->m_nNumofCachedHessianRow;
			//m_pHessianOps->ReadHessianFullRow(pFile, nIndexInSSD, 1, pfHessianFullRow);
			//memcpy(pfSVHessianSubRow, pfHessianFullRow + nStartPos, nNumofTestSamples * sizeof(float_point));
		}

		for(int j = 0; j < nNumofTestSamples; j++)
		{
			//store kernel values in a matrix with the form that row is testing samples, column is SVs.
			pfSVsKernelValues[j * (long long)nNumofSVs + i] = pfSVHessianSubRow[j];
		}
	}
	fclose(pFile);
	delete[] pfSVHessianSubRow;
	delete[] pfHessianFullRow;
}

/*
 * @brief: read kernel values based on testing examples
 */
void CSVMPredictor::ReadKVbasedOnTest(real *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs, int nNumofTestSamples)
{
	FILE *pFile = fopen(HESSIAN_FILE, "rb");
	real *pfSVHessianSubRow = new real[nNumofSVs];
	memset(pfSVHessianSubRow, 0, sizeof(real) * nNumofSVs);

	real *pfHessianRow = new real[m_pHessianReader->m_nTotalNumofInstance];

	int nTestStartId = m_nTestStart;
	assert(nTestStartId >= 0);
	int nTestEndId = nTestStartId + nNumofTestSamples - 1;//include the last sample

	for(int i = nTestStartId; i <= nTestEndId; i++)
	{
		//read part of the Hessian Row
		//if the hessian row is in host memory
		if(m_pHessianReader->m_nNumofCachedHessianRow > i)
		{
			for(int j = 0; j < nNumofSVs; j++)
			{
				pfSVHessianSubRow[j] = m_pHessianReader->m_pfHessianRowsInHostMem[i * (long long)m_pHessianReader->m_nTotalNumofInstance + pnSVSampleId[j]];
			}
		}
		else//the hessian row is in SSD
		{
			m_pHessianReader->ReadRow(i, pfHessianRow);
			for(int j = 0; j < nNumofSVs; j++)
			{
				pfSVHessianSubRow[j] = pfHessianRow[pnSVSampleId[j]];
			}
		}

		for(int j = 0; j < nNumofSVs; j++)
		{
			//store kernel values in a matrix with the form that row is testing samples, column is SVs.
			pfSVsKernelValues[(i - nTestStartId) * (long long)nNumofSVs + j] = pfSVHessianSubRow[j];
		}
	}
	if(pFile != NULL)
		fclose(pFile);
	delete[] pfSVHessianSubRow;
	delete[] pfHessianRow;
}

/**
 * @brief: read kernel values from precomputed results
 */
void CSVMPredictor::ReadFromHessian(real *pfSVsKernelValues, int *pnSVSampleId, int nNumofSVs,
									int *pnTestSampleId, int nNumofTestSamples)
{
	//get Hessian rows of support vectors
	m_pHessianReader->AllocateBuffer(1);
	if(nNumofSVs >= nNumofTestSamples)
	{
		m_pHessianReader->SetInvolveData(-1, -1, 0, m_pHessianReader->m_nTotalNumofInstance - 1);
		ReadKVbasedOnTest(pfSVsKernelValues, pnSVSampleId, nNumofSVs, nNumofTestSamples);
	}
	else
	{
		m_pHessianReader->SetInvolveData(-1, -1, pnTestSampleId[0], pnTestSampleId[nNumofTestSamples - 1]);
		ReadKVbasedOnSV(pfSVsKernelValues, pnSVSampleId, nNumofSVs, nNumofTestSamples);
	}
	m_pHessianReader->ReleaseBuffer();
}

/**
 * @brief: allocate memory for kernel values involved in the prediction
 */
real *CSVMPredictor::AllocateKVMem(int nNumofSVs, const int &nNumofTestSamples)
{
	//store sub Hessian Matrix
	real *pfSVsKernelValues = new real[nNumofTestSamples * nNumofSVs];
	memset(pfSVsKernelValues, 0, sizeof(real) * nNumofTestSamples * nNumofSVs);
	return pfSVsKernelValues;
}

/**
 * @return the number of support vectors in the model
 */
int CSVMPredictor::GetNumSV(svm_model *pModel)
{
	return (pModel->nSV[0] + pModel->nSV[1]);
}

/**
 * @brief: predict the label helper function
 */
real* CSVMPredictor::PredictLabel(svm_model *pModel, int nNumofTestSamples, real *pfSVsKernelValues)
{
	//get infomation from SVM model
	int nNumofSVs = GetNumSV(pModel);
	real fBias = *(pModel->rho);
	real **pyfSVsYiAlpha = pModel->sv_coef;
	real *pfSVsYiAlpha = pyfSVsYiAlpha[0];
	int *pnSVsLabel = pModel->label;
	real *pfYiAlphaofSVs;

	/*compute y_i*alpha_i*K(i, z) by GPU, where i is id of support vector.
	 * pfDevSVYiAlphaHessian stores in the order of T1 sv1 sv2 ... T2 sv1 sv2 ... T3 sv1 sv2 ...
	 */
	real *pfDevSVYiAlphaHessian;
	real *pfDevSVsYiAlpha;
	int *pnDevSVsLabel;

	//if the memory is not enough for the storage when classifying all testing samples at once, divide it into multiple parts

	StorageManager *manager = StorageManager::getManager();
	int nMaxNumofFloatPoint = manager->GetFreeGPUMem();
	int nNumofPart = Ceil(nNumofSVs * nNumofTestSamples, nMaxNumofFloatPoint);

//	cout << "cache size is: " << nMaxNumofFloatPoint << " v.s.. " << nNumofSVs * nNumofTestSamples << endl;
//	cout << "perform classification in " << nNumofPart << " time(s)" << endl;

	//allocate memory for storing classification result
	real *pfClassificaitonResult = new real[nNumofTestSamples];
	//initialise the size of each part
	int *pSizeofPart = new int[nNumofPart];
	int nAverageSize = nNumofTestSamples / nNumofPart;
	for(int i = 0; i < nNumofPart; i++)
	{
		if(i != nNumofPart - 1)
		{
			pSizeofPart[i] = nAverageSize;
		}
		else
		{
			pSizeofPart[i] = nNumofTestSamples - nAverageSize * i;
		}
	}

	//perform classification for each part
	for(int i = 0; i < nNumofPart; i++)
	{
        checkCudaErrors(cudaMalloc((void**)&pfDevSVYiAlphaHessian, sizeof(real) * nNumofSVs * pSizeofPart[i]));
        checkCudaErrors(cudaMalloc((void**)&pfDevSVsYiAlpha, sizeof(real) * nNumofSVs));
        checkCudaErrors(cudaMalloc((void**)&pnDevSVsLabel, sizeof(int) * nNumofSVs));

        checkCudaErrors(cudaMemset(pfDevSVYiAlphaHessian, 0, sizeof(real) * nNumofSVs * pSizeofPart[i]));
        checkCudaErrors(cudaMemset(pfDevSVsYiAlpha, 0, sizeof(real) * nNumofSVs));
        checkCudaErrors(cudaMemset(pnDevSVsLabel, 0, sizeof(int) * nNumofSVs));

        checkCudaErrors(cudaMemcpy(pfDevSVYiAlphaHessian, pfSVsKernelValues + i * nAverageSize * nNumofSVs,
                                   sizeof(real) * nNumofSVs * pSizeofPart[i], cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pfDevSVsYiAlpha, pfSVsYiAlpha, sizeof(real) * nNumofSVs, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(pnDevSVsLabel, pnSVsLabel, sizeof(int) * nNumofSVs, cudaMemcpyHostToDevice));

        //compute y_i*alpha_i*K(i, z)
        int nVecMatxMulGridDimY = pSizeofPart[i];
        int nVecMatxMulGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
        dim3 vecMatxMulGridDim(nVecMatxMulGridDimX, nVecMatxMulGridDimY);
        VectorMatrixMul<<<vecMatxMulGridDim, BLOCK_SIZE>>>(pfDevSVsYiAlpha, pfDevSVYiAlphaHessian, pSizeofPart[i], nNumofSVs);

        //perform classification
        ComputeClassLabel(pSizeofPart[i], pfDevSVYiAlphaHessian,
					  nNumofSVs, fBias, pfClassificaitonResult + i * nAverageSize);

        if(pfClassificaitonResult == NULL)
        {
            cerr << "error in computeSVYiAlphaHessianSum" << endl;
            exit(-1);
        }


        //free memory
        checkCudaErrors(cudaFree(pfDevSVYiAlphaHessian));
        pfDevSVYiAlphaHessian = NULL;
        checkCudaErrors(cudaFree(pfDevSVsYiAlpha));
        checkCudaErrors(cudaFree(pnDevSVsLabel));
	}

	return pfClassificaitonResult;
}

/*
 * @brief: predict class labels using precomputed kernel valules
 */
real* CSVMPredictor::Predict(svm_model *pModel, int *pnTestSampleId, const int &nNumofTestSamples)
{
	real *pfReturn = NULL;
	if(pModel == NULL)
	{
		cerr << "error in Predict function: invalid input params" << endl;
		return pfReturn;
	}

	//get infomation from SVM model
	int nNumofSVs = GetNumSV(pModel);
	int *pnSVSampleId = pModel->pnIndexofSV;

	//store sub Hessian Matrix
	real *pfSVsKernelValues = AllocateKVMem(nNumofSVs, nNumofTestSamples);

	//get Hessian rows of support vectors
	ReadFromHessian(pfSVsKernelValues, pnSVSampleId, nNumofSVs, pnTestSampleId, nNumofTestSamples);

	pfReturn = PredictLabel(pModel, nNumofTestSamples, pfSVsKernelValues);

	delete[] pfSVsKernelValues;

	return pfReturn;
}

double k_function(const svm_node *x, const svm_node *y,
			  const SVMParam &param)
{
	switch(param.kernel_type)
	{
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1)
			{
				if(x->index == y->index)
				{
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}

			return exp(-param.gamma*sum);
		}
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable
	}
}

/**
 * @brief: compute kernel values on-the-fly
 */
void CSVMPredictor::ComputeOnTheFly(real *pfSVsKernelValues, svm_model *model, svm_node **pInstance, int numInstance)
{
	int nr_class = model->nr_class;
	int l = model->l;

	//store kernel values in a matrix with the form that row is testing samples, column is SVs.
	for(int j = 0; j < numInstance; j++)
	{
		for(int i=0;i<l;i++)
		{
			pfSVsKernelValues[j*l+i] = k_function(pInstance[j], model->SV[i], model->param);
		}
	}
}

/**
 * @brief: predict labels with computing kernel values on-the-fly
 */
real* CSVMPredictor::Predict(svm_model *pModel, svm_node **pInstance, int numInstance)
{
	real *pfReturn = NULL;
	if(pModel == NULL)
	{
		cerr << "error in Predict function: invalid input params" << endl;
		return pfReturn;
	}

	//get infomation from SVM model
	int nNumofSVs = GetNumSV(pModel);

	//store sub Hessian Matrix
	real *pfSVsKernelValues = AllocateKVMem(nNumofSVs, numInstance);

	//get Hessian rows of support vectors
	ComputeOnTheFly(pfSVsKernelValues, pModel, pInstance, numInstance);

	pfReturn = PredictLabel(pModel, numInstance, pfSVsKernelValues);

	delete[] pfSVsKernelValues;

	return pfReturn;
}


/*
 * @brief: compute/predict the labels of testing samples
 * @output: a set of class labels, associated to testing samples
 */
real* CSVMPredictor::ComputeClassLabel(int nNumofTestSamples,
									  real *pfDevSVYiAlphaHessian, const int &nNumofSVs,
									  real fBias, real *pfFinalResult)
{
	real *pfReturn = NULL;
	if(nNumofTestSamples <= 0 ||
	   pfDevSVYiAlphaHessian == NULL ||
	   nNumofSVs <= 0)
	{
		cerr << "error in computeSVYiAlphaHessianSum: invalid input params" << endl;
		return pfReturn;
	}

	//compute the size of current processing testing samples
    long long lMega = 1024 * 1024;
    long long cacheSizeInByte = (CACHE_SIZE * lMega * 4);
	long long nMaxSizeofProcessingSample = (cacheSizeInByte / (sizeof(real) * nNumofSVs));

	//reduce by half
	nMaxSizeofProcessingSample = nMaxSizeofProcessingSample / 2;

	//if the number of samples in small
	if(nMaxSizeofProcessingSample > nNumofTestSamples)
	{
		nMaxSizeofProcessingSample = nNumofTestSamples;
	}
	//compute grid size, and block size for partial sum
	int nPartialGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
	int nPartialGridDimY = nMaxSizeofProcessingSample;
	dim3 dimPartialSumGrid(nPartialGridDimX, nPartialGridDimY);
	dim3 dimPartialSumBlock(BLOCK_SIZE);

	//compute grid size, and block size for global sum and class label computing
	int nGlobalGridDimX = 1;
	int nGlobalGridDimY = nMaxSizeofProcessingSample;
	dim3 dimGlobalSumGrid(nGlobalGridDimX, nGlobalGridDimY); //can use 1D grid
	dim3 dimGlobalSumBlock(nPartialGridDimX);

	//memory for computing partial sum by GPU
	real* pfDevPartialSum;
//	cout << "dimx=" << nPartialGridDimX << "; dimy=" << nPartialGridDimY << endl;
	checkCudaErrors(cudaMalloc((void**)&pfDevPartialSum, sizeof(real) * nPartialGridDimX * nPartialGridDimY));
	checkCudaErrors(cudaMemset(pfDevPartialSum, 0, sizeof(real) * nPartialGridDimX * nPartialGridDimY));

	//memory for computing global sum by GPU
	real *pfDevClassificationResult;
	checkCudaErrors(cudaMalloc((void**)&pfDevClassificationResult, sizeof(real) * nGlobalGridDimY));
	checkCudaErrors(cudaMemset(pfDevClassificationResult, 0, sizeof(real) * nGlobalGridDimY));

	//reduce step size of partial sum, and global sum
	int nPartialReduceStepSize = 0;
	nPartialReduceStepSize = (int)pow(2, (ceil(log2((float)BLOCK_SIZE))-1));
	int nGlobalReduceStepSize = 0;
	nGlobalReduceStepSize = (int) pow(2, ceil(log2((float) nPartialGridDimX)) - 1);

	for(int nStartPosofTestSample = 0; nStartPosofTestSample < nNumofTestSamples; nStartPosofTestSample += nMaxSizeofProcessingSample)
	{
		if(nStartPosofTestSample + nMaxSizeofProcessingSample > nNumofTestSamples)
		{
			//the last part of the testing samples
			nMaxSizeofProcessingSample = nNumofTestSamples - nStartPosofTestSample;
			nPartialGridDimY = nMaxSizeofProcessingSample;
			dimPartialSumGrid = dim3(nPartialGridDimX, nPartialGridDimY);
			nGlobalGridDimY = nMaxSizeofProcessingSample;
			dimGlobalSumGrid = dim3(nGlobalGridDimX, nGlobalGridDimY);

			checkCudaErrors(cudaFree(pfDevPartialSum));
			checkCudaErrors(cudaMalloc((void**)&pfDevPartialSum, sizeof(real) * nPartialGridDimX * nPartialGridDimY));
			checkCudaErrors(cudaMemset(pfDevPartialSum, 0, sizeof(real) * nPartialGridDimX * nPartialGridDimY));

			checkCudaErrors(cudaFree(pfDevClassificationResult));
			checkCudaErrors(cudaMalloc((void**)&pfDevClassificationResult, sizeof(real) * nGlobalGridDimY));
			checkCudaErrors(cudaMemset(pfDevClassificationResult, 0, sizeof(real) * nGlobalGridDimY));
		}
		/********* compute partial sum **********/
		ComputeKernelPartialSum<<<dimPartialSumGrid, dimPartialSumBlock, BLOCK_SIZE * sizeof(real)>>>
							   (pfDevSVYiAlphaHessian, nNumofSVs, pfDevPartialSum,
								nPartialReduceStepSize);

		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			cerr << "cuda error in computeSVYiAlphaHessianSum: failed at ComputePartialSum: " << cudaGetErrorString(error) << endl;
            exit(-1);
		}

		/********** compute global sum and class label *********/
		//compute global sum
		ComputeKernelGlobalSum<<<dimGlobalSumGrid, dimGlobalSumBlock, nPartialGridDimX * sizeof(real)>>>
							  (pfDevClassificationResult, fBias,
							   pfDevPartialSum, nGlobalReduceStepSize);
		cudaDeviceSynchronize();

		error = cudaGetLastError();
		if(error != cudaSuccess)
		{
			cerr << "cuda error in computeSVYiAlphaHessianSum: failed at ComputeGlobalSum: " << cudaGetErrorString(error) << endl;
			exit(-1);
		}

		//copy classification result back
		checkCudaErrors(cudaMemcpy(pfFinalResult + nStartPosofTestSample, pfDevClassificationResult,
								 nMaxSizeofProcessingSample * sizeof(real), cudaMemcpyDeviceToHost));
	}

	checkCudaErrors(cudaFree(pfDevPartialSum));
	checkCudaErrors(cudaFree(pfDevClassificationResult));

	pfReturn = pfFinalResult;
	return pfReturn;
}

/*
 * @brief: set data involved in prediction
 */
bool CSVMPredictor::SetInvolvePredictionData(int nStart1, int nEnd1)
{
	bool bReturn = false;
	m_nTestStart = nStart1;
	bReturn = m_pHessianReader->SetInvolveData(-1, -1, 0, m_pHessianReader->m_nTotalNumofInstance - 1);

	return bReturn;
}
