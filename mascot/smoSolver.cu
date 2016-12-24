
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
	hessianDiag = new float_point[nNumofInstance];
	memset(hessianDiag, 0, sizeof(float_point) * nNumofInstance);
	string strHessianDiagFile = HESSIAN_DIAG_FILE;
	bool bReadDiag = m_pHessianReader->GetHessianDiag(strHessianDiagFile, nNumofInstance, hessianDiag);

	//allocate memory for Hessian diagonal
	checkCudaErrors(cudaMalloc((void**)&devHessianDiag, sizeof(float_point) * nNumofInstance));
	checkCudaErrors(cudaMemset(devHessianDiag, 0, sizeof(float_point) * nNumofInstance));
	//copy Hessian diagonal from CPU to GPU
	checkCudaErrors(cudaMemcpy(devHessianDiag, hessianDiag, sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice));
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
	GetBlockMinYiGValue<<<gridSize, BLOCK_SIZE>>>(pfDevYiFValue, pfDevAlpha, pnDevLabel, gfPCost,
												 nNumofTrainingSamples, devBlockMin, devBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMin, devBlockMinGlobalKey, numOfBlock, pfDevYiFValue, NULL, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
	IdofInstanceOne = (int)hostBuffer[0];
	float_point fMinValue;
	fMinValue = hostBuffer[1];
	devHessianInstanceRow1 = GetHessianRow(nNumofTrainingSamples,	IdofInstanceOne);

	//lock cached entry for the sample one, in case it is replaced by sample two
	m_pGPUCache->LockCacheEntry(IdofInstanceOne);

	float_point fUpSelfKernelValue = 0;
	fUpSelfKernelValue = hessianDiag[IdofInstanceOne];
	//select second sample

	upValue = -fMinValue;

	//get block level min (-b_ij*b_ij/a_ij)
	GetBlockMinLowValue<<<gridSize, BLOCK_SIZE>>>
					   (pfDevYiFValue, pfDevAlpha, pnDevLabel, gfNCost, nNumofTrainingSamples, devHessianDiag,
						devHessianInstanceRow1, upValue, fUpSelfKernelValue, devBlockMin, devBlockMinGlobalKey,
						devBlockMinYiGValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE>>>
				(devBlockMin, devBlockMinGlobalKey,
				 numOfBlock, pfDevYiFValue, devHessianInstanceRow1, devBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMinYiGValue, numOfBlock, devBuffer);

//	cudaThreadSynchronize();
	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
	IdofInstanceTwo = int(hostBuffer[0]);

	//get kernel value K(Sample1, Sample2)
	float_point fKernelValue = 0;
	float_point fMinLowValue;
	fMinLowValue = hostBuffer[1];
	fKernelValue = hostBuffer[2];


	devHessianInstanceRow2 = GetHessianRow(nNumofTrainingSamples,	IdofInstanceTwo);
//	cudaDeviceSynchronize();


	m_fLowValue = -hostBuffer[3];
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
	hostBuffer[0] = IdofInstanceOne;
	hostBuffer[1] = fAlpha1;
	hostBuffer[2] = IdofInstanceTwo;
	hostBuffer[3] = fAlpha2;
	cudaMemcpy(devBuffer, hostBuffer, sizeof(float_point) * 4, cudaMemcpyHostToDevice);
	UpdateYiFValueKernel<<<gridSize, BLOCK_SIZE>>>(pfDevAlpha, devBuffer, pfDevYiFValue,
												  devHessianInstanceRow1, devHessianInstanceRow2,
												  fY1AlphaDiff, fY2AlphaDiff, nNumofTrainingSamples);

	return 0;
}
