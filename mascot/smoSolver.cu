
#include "../svm-shared/smoSolver.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <sys/time.h>


/*
 * @brief: initialize cache for training SVM
 * @param: nCacheSize: the size of cache
 * @param: nNumofTrainingSamples: the number of training samples
 * @param: nNumofElementEachRowInCache: the number of elements of each row in cache
 */
bool CSMOSolver::InitCache(const int &nCacheSize, const int &nNumofInstance)
{
	bool bReturn = true;
	//check input parameters
	if(nCacheSize < 0 || nNumofInstance <= 0)
	{
		bReturn = false;
		cerr << "error in InitCache: invalid input param" << endl;
		return bReturn;
	}

	//allocate memory for Hessian rows caching in GPU with memory alignment
	cout << "Allocate GPU cache memory: number of ins is " << nNumofInstance << " and cache size is " << nCacheSize << endl;
	size_t sizeOfEachRowInCache;
	checkCudaErrors(cudaMallocPitch((void**)&m_pfDevHessianMatrixCache, &sizeOfEachRowInCache, nNumofInstance * sizeof(float_point), nCacheSize));
	//temp memory for reading result to cache
	m_lNumofElementEachRowInCache = sizeOfEachRowInCache / sizeof(float_point);
	if(m_lNumofElementEachRowInCache != nNumofInstance)
	{
		cout << "cache memory aligned to: " << m_lNumofElementEachRowInCache
			 << "; number of the training instances is: " << nNumofInstance << endl;
	}

	long long nSpaceForCache = (long long)nCacheSize * m_lNumofElementEachRowInCache;
	checkCudaErrors(cudaMemset(m_pfDevHessianMatrixCache, 0, sizeof(float_point) * nSpaceForCache));

	//initialize cache
	m_pGPUCache->SetCacheSize(nCacheSize);
	m_pGPUCache->InitializeCache(nCacheSize, nNumofInstance);

	//read Hessian diagonal
	m_pfGValue = new float_point[nNumofInstance];
	m_pfAlpha = new float_point[nNumofInstance];
	m_pfDiagHessian = new float_point[nNumofInstance];
	memset(m_pfDiagHessian, 0, sizeof(float_point) * nNumofInstance);
	string strHessianDiagFile = HESSIAN_DIAG_FILE;
	bool bReadDiag = m_pHessianReader->GetHessianDiag(strHessianDiagFile, nNumofInstance, m_pfDiagHessian);

	//allocate memory for Hessian diagonal
	checkCudaErrors(cudaMalloc((void**)&m_pfDevDiagHessian, sizeof(float_point) * nNumofInstance));
	checkCudaErrors(cudaMemset(m_pfDevDiagHessian, 0, sizeof(float_point) * nNumofInstance));
	//copy Hessian diagonal from CPU to GPU
	checkCudaErrors(cudaMemcpy(m_pfDevDiagHessian, m_pfDiagHessian, sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice));
	assert(cudaGetLastError() == cudaSuccess);


	return bReturn;
}


long nTimeOfUpdateAlpha = 0;
long nTimeOfSelect1stSample = 0;
long nTimeOfSelect2ndSample = 0;
long nTimeOfUpdateYiFValue = 0;
long nTimeofGetHessian = 0;
/*
 * @brief: this is a function which contains all the four steps
 */
int CSMOSolver::Iterate(float_point *pfDevYiFValue, float_point *pfDevAlpha, int *pnDevLabel, const int &nNumofTrainingSamples)
{
//	cudaDeviceSynchronize();
	//block level reducer
	GetBlockMinYiGValue<<<dimGridThinThread, BLOCK_SIZE>>>(pfDevYiFValue, pfDevAlpha, pnDevLabel, gfPCost,
												 nNumofTrainingSamples, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey, m_nNumofBlock, pfDevYiFValue, NULL, m_pfDevBuffer);

	//copy result back to host
	cudaMemcpy(m_pfHostBuffer, m_pfDevBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
	IdofInstanceOne = (int)m_pfHostBuffer[0];
	float_point fMinValue;
	fMinValue = m_pfHostBuffer[1];
	m_pfDevHessianSampleRow1 = GetHessianRow(nNumofTrainingSamples,	IdofInstanceOne);

	//lock cached entry for the sample one, in case it is replaced by sample two
	m_pGPUCache->LockCacheEntry(IdofInstanceOne);

	float_point fUpSelfKernelValue = 0;
	fUpSelfKernelValue = m_pfDiagHessian[IdofInstanceOne];
	//select second sample

	upValue = -fMinValue;

	//get block level min (-b_ij*b_ij/a_ij)
	GetBlockMinLowValue<<<dimGridThinThread, BLOCK_SIZE>>>
					   (pfDevYiFValue, pfDevAlpha, pnDevLabel, gfNCost, nNumofTrainingSamples, m_pfDevDiagHessian,
						m_pfDevHessianSampleRow1, upValue, fUpSelfKernelValue, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
						m_pfDevBlockMinYiFValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE>>>
				(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
				 m_nNumofBlock, pfDevYiFValue, m_pfDevHessianSampleRow1, m_pfDevBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(m_pfDevBlockMinYiFValue, m_nNumofBlock, m_pfDevBuffer);

//	cudaThreadSynchronize();
	//copy result back to host
	cudaMemcpy(m_pfHostBuffer, m_pfDevBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
	IdofInstanceTwo = int(m_pfHostBuffer[0]);

	//get kernel value K(Sample1, Sample2)
	float_point fKernelValue = 0;
	float_point fMinLowValue;
	fMinLowValue = m_pfHostBuffer[1];
	fKernelValue = m_pfHostBuffer[2];


	m_pfDevHessianSampleRow2 = GetHessianRow(nNumofTrainingSamples,	IdofInstanceTwo);
//	cudaDeviceSynchronize();


	m_fLowValue = -m_pfHostBuffer[3];
	//check if the problem is converged
	if(upValue + m_fLowValue <= EPS)
	{
		//cout << m_fUpValue << " : " << m_fLowValue << endl;
		//m_pGPUCache->PrintCachingStatistics();
		return 1;
	}

	float_point fY1AlphaDiff, fY2AlphaDiff;
	UpdateTwoWeight(fMinLowValue, fMinValue, IdofInstanceOne, IdofInstanceTwo, fKernelValue,
					fY1AlphaDiff, fY2AlphaDiff);
	float_point fAlpha1 = m_pfAlpha[IdofInstanceOne];
	float_point fAlpha2 = m_pfAlpha[IdofInstanceTwo];

	m_pGPUCache->UnlockCacheEntry(IdofInstanceOne);

	//update yiFvalue
	//copy new alpha values to device
	m_pfHostBuffer[0] = IdofInstanceOne;
	m_pfHostBuffer[1] = fAlpha1;
	m_pfHostBuffer[2] = IdofInstanceTwo;
	m_pfHostBuffer[3] = fAlpha2;
	cudaMemcpy(m_pfDevBuffer, m_pfHostBuffer, sizeof(float_point) * 4, cudaMemcpyHostToDevice);
	UpdateYiFValueKernel<<<dimGridThinThread, BLOCK_SIZE>>>(pfDevAlpha, m_pfDevBuffer, pfDevYiFValue,
												  m_pfDevHessianSampleRow1, m_pfDevHessianSampleRow2,
												  fY1AlphaDiff, fY2AlphaDiff, nNumofTrainingSamples);

	return 0;
}
