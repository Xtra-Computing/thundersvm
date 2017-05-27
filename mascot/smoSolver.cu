
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
	checkCudaErrors(cudaMallocPitch((void**)&m_pfDevHessianMatrixCache, &sizeOfEachRowInCache, nNumofInstance * sizeof(real), nCacheSize));
	//temp memory for reading result to cache
	m_lNumofElementEachRowInCache = sizeOfEachRowInCache / sizeof(real);
	if(m_lNumofElementEachRowInCache != nNumofInstance)
	{
		cout << "cache memory aligned to: " << m_lNumofElementEachRowInCache
			 << "; number of the training instances is: " << nNumofInstance << endl;
	}

	long long nSpaceForCache = (long long)nCacheSize * m_lNumofElementEachRowInCache;
	checkCudaErrors(cudaMemset(m_pfDevHessianMatrixCache, 0, sizeof(real) * nSpaceForCache));

	//initialize cache
	m_pGPUCache->SetCacheSize(nCacheSize);
	m_pGPUCache->InitializeCache(nCacheSize, nNumofInstance);

	//read Hessian diagonal
	m_pfGValue = new real[nNumofInstance];

	string strHessianDiagFile = HESSIAN_DIAG_FILE;
	bool bReadDiag = m_pHessianReader->GetHessianDiag(strHessianDiagFile, nNumofInstance, hessianDiag);

	//copy Hessian diagonal from CPU to GPU
	checkCudaErrors(cudaMemcpy(devHessianDiag, hessianDiag, sizeof(real) * nNumofInstance, cudaMemcpyHostToDevice));
	assert(cudaGetLastError() == cudaSuccess);

	return bReturn;
}

/*
 * @brief: this is a function which contains all the four steps
 */
int CSMOSolver::Iterate(real *pfDevYiFValue, real *pfDevAlpha, int *pnDevLabel, const int &nNumofTrainingSamples)
{
	//variables used in search
	devYiGValue = pfDevYiFValue;
	devAlpha = pfDevAlpha;
	devLabel = pnDevLabel;
	SelectFirst(nNumofTrainingSamples, gfPCost);
   
	//lock cached entry for the sample one, in case it is replaced by sample two

	m_pGPUCache->LockCacheEntry(IdofInstanceOne);
    SelectSecond(nNumofTrainingSamples, gfNCost);
	
	IdofInstanceTwo = int(hostBuffer[0]);

	//get kernel value K(Sample1, Sample2)
	real fKernelValue = 0;
	real fMinLowValue;
	fMinLowValue = hostBuffer[1];
	fKernelValue = hostBuffer[2];

	devHessianInstanceRow2 = GetHessianRow(nNumofTrainingSamples,	IdofInstanceTwo);

	m_fLowValue = -hostBuffer[3];
	//check if the problem is converged
	if(upValue + m_fLowValue <= EPS)
	{
		//cout << m_fUpValue << " : " << m_fLowValue << endl;
		//m_pGPUCache->PrintCachingStatistics();
		return 1;
	}

	real fY1AlphaDiff, fY2AlphaDiff;
	UpdateTwoWeight(fMinLowValue, -upValue, IdofInstanceOne, IdofInstanceTwo, fKernelValue,
					fY1AlphaDiff, fY2AlphaDiff, m_pnLabel, gfPCost);

	m_pGPUCache->UnlockCacheEntry(IdofInstanceOne);

	UpdateYiGValue(nNumofTrainingSamples, fY1AlphaDiff, fY2AlphaDiff);
    return 0;
}
