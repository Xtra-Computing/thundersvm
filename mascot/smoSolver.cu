
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
	checkCudaErrors(cudaMemsetAsync(m_pfDevHessianMatrixCache, 0, sizeof(float_point) * nSpaceForCache, stream));

	//initialize cache
	m_pGPUCache->SetCacheSize(nCacheSize);
	m_pGPUCache->InitializeCache(nCacheSize, nNumofInstance);

	//read Hessian diagonal
//	m_pfGValue = new float_point[nNumofInstance];
//	m_pfAlpha = new float_point[nNumofInstance];
//	m_pfDiagHessian = new float_point[nNumofInstance];
    checkCudaErrors(cudaMallocHost((void**)&m_pfGValue,sizeof(float_point)*nNumofInstance));
    checkCudaErrors(cudaMallocHost((void**)&m_pfAlpha,sizeof(float_point)*nNumofInstance));
    checkCudaErrors(cudaMallocHost((void**)&m_pfDiagHessian,sizeof(float_point)*nNumofInstance));
	memset(m_pfDiagHessian, 0, sizeof(float_point) * nNumofInstance);
	string strHessianDiagFile = HESSIAN_DIAG_FILE;
	bool bReadDiag = m_pHessianReader->GetHessianDiag(strHessianDiagFile, nNumofInstance, m_pfDiagHessian);

	//allocate memory for Hessian diagonal
	checkCudaErrors(cudaMalloc((void**)&m_pfDevDiagHessian, sizeof(float_point) * nNumofInstance));
	checkCudaErrors(cudaMemsetAsync(m_pfDevDiagHessian, 0, sizeof(float_point) * nNumofInstance, stream));
	//copy Hessian diagonal from CPU to GPU
	checkCudaErrors(cudaMemcpyAsync(m_pfDevDiagHessian, m_pfDiagHessian, sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice, stream));
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
	//####### timer
	timeval tSelect1S, tSelect1E;
	gettimeofday(&tSelect1S, NULL);

/*	cudaProfilerStart();
	//block level reducer
*/
    size_t sharedMemSize = (sizeof(int) + sizeof(float_point)) * BLOCK_SIZE;
    GetBlockMinYiGValue<<<dimGridThinThread, BLOCK_SIZE,sharedMemSize,stream>>>(pfDevYiFValue, pfDevAlpha, pnDevLabel, gfPCost,
												 nNumofTrainingSamples, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE,sharedMemSize,stream>>>(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey, m_nNumofBlock, pfDevYiFValue, NULL, m_pfDevBuffer);
//	cudaStreamSynchronize(stream);

	//when the number of blocks is larger than 65535, compute the number of grid
/*	int nNumofBlock = Ceil(nNumofTrainingSamples, (BLOCK_SIZE * TASK_OF_THREAD));
	int nGridDimY = 0;
	nGridDimY = Ceil(nNumofBlock, NUM_OF_BLOCK);

	int nGridDimX = 0;
	if(nNumofBlock > NUM_OF_BLOCK)
		nGridDimX = NUM_OF_BLOCK;
	else
		nGridDimX = nNumofBlock;
	dim3 temp(nGridDimX, nGridDimY);
	//cout << nGridDimX << " y: " << nGridDimY << endl;
	GetBigBlockMinYiGValue<<<temp, BLOCK_SIZE>>>(pfDevYiFValue, pfDevAlpha, pnDevLabel, gfPCost,
												 nNumofTrainingSamples, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey, nNumofBlock, pfDevYiFValue, NULL, m_pfDevBuffer);
*/
	//copy result back to host
	cudaMemcpyAsync(m_pfHostBuffer, m_pfDevBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost,stream);
	cudaStreamSynchronize(stream);
//    cudaDeviceSynchronize();
	/*error = cudaGetLastError();
	assert(error == cudaSuccess);*/
	m_nIndexofSampleOne = (int)m_pfHostBuffer[0];
	//cout << m_nIndexofSampleOne << endl;
	float_point fMinValue;
	fMinValue = m_pfHostBuffer[1];

	//######## timer
	gettimeofday(&tSelect1E, NULL);
	nTimeOfSelect1stSample += (tSelect1E.tv_sec - tSelect1S.tv_sec) * 1000 * 1000;
	nTimeOfSelect1stSample += (tSelect1E.tv_usec - tSelect1S.tv_usec);

	timeval tGetHessian1S, tGetHessian1E;
	gettimeofday(&tGetHessian1S, NULL);
	m_pfDevHessianSampleRow1 = GetHessianRow(nNumofTrainingSamples,	m_nIndexofSampleOne);
	cudaDeviceSynchronize();
	gettimeofday(&tGetHessian1E, NULL);
	nTimeofGetHessian += (tGetHessian1E.tv_sec - tGetHessian1S.tv_sec) * 1000 * 1000;
	nTimeofGetHessian += (tGetHessian1E.tv_usec - tGetHessian1S.tv_usec);

	//lock cached entry for the sample one, in case it is replaced by sample two
	m_pGPUCache->LockCacheEntry(m_nIndexofSampleOne);
	//assert(m_pfDevHessianSampleRow1 != NULL);
	//assert(m_nIndexofSampleOne >= 0);

	float_point fUpSelfKernelValue = 0;
	fUpSelfKernelValue = m_pfDiagHessian[m_nIndexofSampleOne];
	//select second sample

	timeval tSelect2S, tSelect2E;
	gettimeofday(&tSelect2S, NULL);
/**/
	m_fUpValue = -fMinValue;
//	cudaStreamSynchronize(m_stream1_Hessian_row);

	//cudaProfilerStart();
	//get block level min (-b_ij*b_ij/a_ij)
/**/	GetBlockMinLowValue<<<dimGridThinThread, BLOCK_SIZE,sharedMemSize,stream>>>
					   (pfDevYiFValue, pfDevAlpha, pnDevLabel, gfNCost, nNumofTrainingSamples, m_pfDevDiagHessian,
						m_pfDevHessianSampleRow1, m_fUpValue, fUpSelfKernelValue, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
						m_pfDevBlockMinYiFValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE,sharedMemSize,stream>>>
				(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
				 m_nNumofBlock, pfDevYiFValue, m_pfDevHessianSampleRow1, m_pfDevBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE,sizeof(float_point)*BLOCK_SIZE,stream>>>(m_pfDevBlockMinYiFValue, m_nNumofBlock, m_pfDevBuffer);

/*	GetBigBlockMinLowValue<<<temp, BLOCK_SIZE>>>
					   (pfDevYiFValue, pfDevAlpha, pnDevLabel, gfNCost, nNumofTrainingSamples, nNumofTrainingSamples, m_pfDevDiagHessian,
						m_pfDevHessianSampleRow1, m_fUpValue, fUpSelfKernelValue, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
						m_pfDevBlockMinYiFValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE>>>
				(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
				 nNumofBlock, pfDevYiFValue, m_pfDevHessianSampleRow1, m_pfDevBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(m_pfDevBlockMinYiFValue, nNumofBlock, m_pfDevBuffer);
*/
//	cudaThreadSynchronize();
	//copy result back to host
	cudaMemcpyAsync(m_pfHostBuffer, m_pfDevBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);
	m_nIndexofSampleTwo = int(m_pfHostBuffer[0]);

//	cout << m_nIndexofSampleOne << " v.s " << m_nIndexofSampleTwo << endl;

	//get kernel value K(Sample1, Sample2)
	float_point fKernelValue = 0;
	float_point fMinLowValue;
	fMinLowValue = m_pfHostBuffer[1];
	fKernelValue = m_pfHostBuffer[2];

	//#### timer
	gettimeofday(&tSelect2E, NULL);
	nTimeOfSelect2ndSample += (tSelect2E.tv_sec - tSelect2S.tv_sec) * 1000 * 1000;
	nTimeOfSelect2ndSample += (tSelect2E.tv_usec - tSelect2S.tv_usec);

	timeval tGetHessian2S, tGetHessian2E;
	gettimeofday(&tGetHessian2S, NULL);
	m_pfDevHessianSampleRow2 = GetHessianRow(nNumofTrainingSamples,	m_nIndexofSampleTwo);
	cudaDeviceSynchronize();
	gettimeofday(&tGetHessian2E, NULL);
	nTimeofGetHessian += (tGetHessian2E.tv_sec - tGetHessian2S.tv_sec) * 1000 * 1000;
	nTimeofGetHessian += (tGetHessian2E.tv_usec - tGetHessian2S.tv_usec);

/*	if(m_nIndexofSampleOne == 11452 && m_nIndexofSampleTwo == 28932)
	{
		PrintGPUHessianRow(m_pfDevHessianSampleRow2, nNumofTrainingSamples);
		exit(0);
	}*/

	m_fLowValue = -m_pfHostBuffer[3];
	//check if the problem is converged
	if(m_fUpValue + m_fLowValue <= EPS)
	{
		//cout << m_fUpValue << " : " << m_fLowValue << endl;
		//m_pGPUCache->PrintCachingStatistics();
		return 1;
	}

	timeval tUpdateAlphaS, tUpdateAlphaE;
	gettimeofday(&tUpdateAlphaS, NULL);
/**/
	float_point fY1AlphaDiff, fY2AlphaDiff;
	UpdateTwoWeight(fMinLowValue, fMinValue, m_nIndexofSampleOne, m_nIndexofSampleTwo, fKernelValue,
					fY1AlphaDiff, fY2AlphaDiff);
	float_point fAlpha1 = m_pfAlpha[m_nIndexofSampleOne];
	float_point fAlpha2 = m_pfAlpha[m_nIndexofSampleTwo];

	m_pGPUCache->UnlockCacheEntry(m_nIndexofSampleOne);

	//update yiFvalue
	//####### timer
	gettimeofday(&tUpdateAlphaE, NULL);

	nTimeOfUpdateAlpha += (tUpdateAlphaE.tv_sec - tUpdateAlphaS.tv_sec) * 1000 * 1000;
	nTimeOfUpdateAlpha += (tUpdateAlphaE.tv_usec - tUpdateAlphaS.tv_usec);
	timeval tUpdateFS, tUpdateFE;
	gettimeofday(&tUpdateFS, NULL);
/**/
	//copy new alpha values to device
	m_pfHostBuffer[0] = m_nIndexofSampleOne;
	m_pfHostBuffer[1] = fAlpha1;
	m_pfHostBuffer[2] = m_nIndexofSampleTwo;
	m_pfHostBuffer[3] = fAlpha2;
	cudaMemcpyAsync(m_pfDevBuffer, m_pfHostBuffer, sizeof(float_point) * 4, cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);
	UpdateYiFValueKernel<<<dimGridThinThread, BLOCK_SIZE,sharedMemSize,stream>>>(pfDevAlpha, m_pfDevBuffer, pfDevYiFValue,
												  m_pfDevHessianSampleRow1, m_pfDevHessianSampleRow2,
												  fY1AlphaDiff, fY2AlphaDiff, nNumofTrainingSamples);

	//####### timer
	gettimeofday(&tUpdateFE, NULL);
	nTimeOfUpdateYiFValue += (tUpdateFE.tv_sec - tUpdateFS.tv_sec) * 1000 * 1000;
	nTimeOfUpdateYiFValue += (tUpdateFE.tv_usec - tUpdateFS.tv_usec);

/*	cudaProfilerStop();
	cudaDeviceReset();
	exit(0);
*/	return 0;
}

void CSMOSolver::setStream(const cudaStream_t &stream) {
    this->stream = stream;
}
