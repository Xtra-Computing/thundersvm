
#include "smoSolver.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include <sys/time.h>

/*
 * @brief: allocate GPU memory for finding block min, which serves as the first sample of the optimized pair
 * @param: nNumofTrainingSamples: the # of training samples in current interation
 * @return: true when memory allocation success
 */
bool CSMOSolver::SMOSolverPreparation(const int &nNumofTrainingSamples)
{
	bool bReturn = true;
	m_nNumofBlock = Ceil(nNumofTrainingSamples, BLOCK_SIZE);

	if(nNumofTrainingSamples <= 0 || m_nNumofBlock <= 0)
	{
		cerr << "error in SeletePairPreparation: invalid input parameters" << endl;
		return false;
	}

	//when the number of blocks is larger than 65535, compute the number of grid
	int nGridDimY = 0;
	nGridDimY = Ceil(m_nNumofBlock, NUM_OF_BLOCK);

	int nGridDimX = 0;
	if(m_nNumofBlock > NUM_OF_BLOCK)
		nGridDimX = NUM_OF_BLOCK;
	else
		nGridDimX = m_nNumofBlock;
	dim3 temp(nGridDimX, nGridDimY);
	dimGridThinThread = temp;

	//allocate device memory
	checkCudaErrors(cudaMalloc((void**)&m_pfDevBlockMin, sizeof(float_point) * m_nNumofBlock));
	checkCudaErrors(cudaMalloc((void**)&m_pnDevBlockMinGlobalKey, sizeof(int) * m_nNumofBlock));
	//for getting maximum low G value
	checkCudaErrors(cudaMalloc((void**)&m_pfDevBlockMinYiFValue, sizeof(float_point) * m_nNumofBlock));

	checkCudaErrors(cudaMalloc((void**)&m_pfDevMinValue, sizeof(float_point)));
	checkCudaErrors(cudaMalloc((void**)&m_pnDevMinKey, sizeof(int)));

	m_pfHostBuffer = new float_point[4];
	checkCudaErrors(cudaMalloc((void**)&m_pfDevBuffer, sizeof(float_point) * 4));//only need 4 float_points

	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "CUDA error occurs at SelectPair" << endl;
		bReturn = false;
	}

	//allocate memory in CPU
	//m_pfHessianRow = new float_point[nNumofTrainingSamples];//for reading hessian row from file
	cudaMallocHost(&m_pfHessianRow, sizeof(float_point) * nNumofTrainingSamples);
	m_pnLabel = new int[nNumofTrainingSamples];

	return bReturn;
}

/*
 * @brief: read a few Hessian rows to GPU cache
 * @param: nNumofTrainingSamples: the number of samples for training
 * @param: sizeOfEachRowInCache: the size of each row in cache. This value is usually larger than nNumofTrainingSamples because of memory alignment
 * @param: pfDevHessianCacheEndPos: GPU cache end position
 */
bool CSMOSolver::ReadToCache(const int &nStartRow, const int &nEndRow, const int &nNumofTrainingSamples, float_point *pfDevHessianCacheEndPos)
{
	bool bReturn = true;

	bool bReadHessian = true;

	int nNumofRows = nEndRow - nStartRow + 1;
	//separate the large batch read into two read
	int nNumofRowsToRead1 = 0;
	int nNumofRowsToRead2 = 0;
	if(nNumofRows < (MAX_SIZE_PER_READ / (nNumofTrainingSamples * sizeof(float_point))))
	{
		nNumofRowsToRead1 = nNumofRows;
	}
	else
	{
		nNumofRowsToRead1 = nNumofRows / 2;
		nNumofRowsToRead2 = nNumofRows - nNumofRowsToRead1;
	}

	int nNumofRowsToRead = nNumofRowsToRead1;
	int nNumofRowsRead = 0;	//the # of rows has been read
	do
	{
		//long long nSpaceForCache = nNumofRowsToRead * nNumofTrainingSamples;
		long long nSpaceForCache = nNumofRowsToRead * m_nNumofElementEachRowInCache;//compute CPU cache size via size of each row in GPU cache due to the memory alignment
		float_point *pfHessianMatrixCache = new float_point[nSpaceForCache];
		memset(pfHessianMatrixCache, 0, sizeof(float_point) * nSpaceForCache);

		//start row and end row position
		int nTempStartRow = nStartRow + nNumofRowsRead;
		int nTempEndRow = nTempStartRow + nNumofRowsToRead - 1;	//as the index of nTempEndRow will be read
		//read Hessian rows
		m_pHessianReader->AllocateBuffer(nTempEndRow - nTempStartRow + 1);
		bReadHessian = m_pHessianReader->ReadHessianRows(m_pFile, nTempStartRow, nTempEndRow, nNumofTrainingSamples, pfHessianMatrixCache, m_nNumofElementEachRowInCache);
		m_pHessianReader->ReleaseBuffer();
		if(bReadHessian == false)
		{
			bReturn = false;
			break;
		}
		//end of cache location on GPU
		float_point *pfDevCacheEndPos = pfDevHessianCacheEndPos + nNumofRowsRead * m_nNumofElementEachRowInCache;
		checkCudaErrors(cudaMemcpy(pfDevCacheEndPos, pfHessianMatrixCache, sizeof(float_point) * nSpaceForCache, cudaMemcpyHostToDevice));
		nNumofRowsRead += nNumofRowsToRead;
		nNumofRowsToRead = nNumofRowsToRead2;
		delete[] pfHessianMatrixCache;
	}while(nNumofRowsRead < nNumofRows);

	return bReturn;
}

/*
 * @brief: initialize cache for training SVM
 * @param: nCacheSize: the size of cache
 * @param: nNumofTrainingSamples: the number of training samples
 * @param: nNumofElementEachRowInCache: the number of elements of each row in cache
 */
bool CSMOSolver::InitCache(const int &nCacheSize, const int &nNumofTrainingSamples)
{
	bool bReturn = true;
	//check input parameters
	if(nCacheSize < 0 || nNumofTrainingSamples <= 0)
	{
		bReturn = false;
		cerr << "error in InitCache: invalid input param" << endl;
		return bReturn;
	}

	//allocate memory for Hessian rows caching in GPU with memory alignment
	size_t sizeOfEachRowInCache;
	checkCudaErrors(cudaMallocPitch((void**)&m_pfDevHessianMatrixCache, &sizeOfEachRowInCache, nNumofTrainingSamples * sizeof(float_point), nCacheSize));
	//temp memory for reading result to cache
	m_nNumofElementEachRowInCache = sizeOfEachRowInCache / sizeof(float_point);
	if(m_nNumofElementEachRowInCache != nNumofTrainingSamples)
	{
//		cout << "memory aligned to: " << m_nNumofElementEachRowInCache << "; number of training samples is: " << nNumofTrainingSamples << endl;
	}
	long long nSpaceForCache = nCacheSize * m_nNumofElementEachRowInCache;
	//checkCudaErrors(cudaMemset(m_pfDevHessianMatrixCache, 0, sizeof(float_point) * nSpaceForCache));

	//initialize cache
	m_pGPUCache->SetCacheSize(nCacheSize);
	m_pGPUCache->InitializeCache(nCacheSize, nNumofTrainingSamples);

	//read Hessian diagonal
	m_pfGValue = new float_point[nNumofTrainingSamples];
	m_pfAlpha = new float_point[nNumofTrainingSamples];
	m_pfDiagHessian = new float_point[nNumofTrainingSamples];
	memset(m_pfDiagHessian, 0, sizeof(float_point) * nNumofTrainingSamples);
	string strHessianDiagFile = HESSIAN_DIAG_FILE;
	bool bReadDiag = m_pHessianReader->GetHessianDiag(strHessianDiagFile, nNumofTrainingSamples, m_pfDiagHessian);

	//allocate memory for Hessian diagonal
	checkCudaErrors(cudaMalloc((void**)&m_pfDevDiagHessian, sizeof(float_point) * nNumofTrainingSamples));
	checkCudaErrors(cudaMemset(m_pfDevDiagHessian, 0, sizeof(float_point) * nNumofTrainingSamples));
	//copy Hessian diagonal from CPU to GPU
	checkCudaErrors(cudaMemcpy(m_pfDevDiagHessian, m_pfDiagHessian, sizeof(float_point) * nNumofTrainingSamples, cudaMemcpyHostToDevice));
	assert(cudaGetLastError() == cudaSuccess);


	return bReturn;
}

/*
 * @brief: release memory used for caching
 */
bool CSMOSolver::CleanCache()
{
	bool bReturn = true;

	//clean cache
	m_pGPUCache->CleanCache();
	checkCudaErrors(cudaFree(m_pfDevHessianMatrixCache));
	checkCudaErrors(cudaFree(m_pfDevDiagHessian));
	checkCudaErrors(cudaFree(m_pfDevBuffer));
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "CUDA error occurs at CleanCache" << endl;
		bReturn = false;
	}

	return bReturn;
}

/*
 * @brief: release memory used by SMO slover
 */
bool CSMOSolver::SMOSolverEnd()
{
	bool bReturn = true;

	//free GPU global memory
	checkCudaErrors(cudaFree(m_pfDevBlockMin));
	checkCudaErrors(cudaFree(m_pfDevBlockMinYiFValue));
	checkCudaErrors(cudaFree(m_pnDevBlockMinGlobalKey));
	checkCudaErrors(cudaFree(m_pfDevMinValue));
	checkCudaErrors(cudaFree(m_pnDevMinKey));
	if(cudaGetLastError() != cudaSuccess)
	{
		cerr << "CUDA error occurs at SelectPair" << endl;
		bReturn = false;
	}

	cudaFreeHost(m_pfHessianRow);
	delete[] m_pnLabel;
	delete[] m_pfDiagHessian;
	delete[] m_pfGValue;
	delete[] m_pfAlpha;

	return bReturn;
}

/*
 * @brief: set data used by SMO solver in Hessian Matrix
 */
bool CSMOSolver::SetInvolveData(int nStart1, int nEnd1, int nStart2, int nEnd2)
{
	bool bReturn = true;

	m_nStart1 = nStart1;
	m_nEnd1 = nEnd1;
	m_nStart2 = nStart2;
	m_nEnd2 = nEnd2;

	bReturn = m_pHessianReader->SetInvolveData(nStart1, nEnd1, nStart2, nEnd2);

	return bReturn;
}


/*
 * @brief: map a given index to Hessian matrix. As in n-fold-cross-validation, data are seperated into n parts.
 */
bool CSMOSolver::MapIndexToHessian(int &nIndex)
{
	bool bReturn = false;
	//check input parameter
	int nTempIndex = nIndex;
	if(nIndex < 0 || (nIndex > m_nEnd1 && nIndex > m_nEnd2))
	{
		cerr << "error in MapIndexToHessian: invalid input parameter" << endl;
		cout << nIndex << " " << m_nEnd1 << " " << m_nEnd2 << endl;
		exit(0);
	}

	bReturn = true;
	if(m_nStart1 != -1)
	{
		if(nIndex <= m_nEnd1)
		{
			return bReturn;
		}
		else
		{
			nTempIndex = nIndex + (m_nStart2 - m_nEnd1 - 1);
			if(nTempIndex < nIndex || nTempIndex > m_nEnd2)
			{
				cerr << "error in MapIndexToHessian" << endl;
				bReturn = false;
			}
		}
	}
	else if(m_nStart2 != -1)
	{
		nTempIndex = nIndex + m_nStart2;
		if(nTempIndex > m_nEnd2)
		{
			cerr << "error in MapIndexToHessian" << endl;
			bReturn = false;
		}
	}
	else
	{
		cerr << "error in MapIndexToHessian: m_nStart1 & 2 equal to -1" << endl;
		bReturn = false;
	}

	nIndex = nTempIndex;
	return bReturn;
}

/*
 * @brief: get a row of Hessian Matrix (the row is either in cache or in secondary memory)
 */
long lGetHessianRowTime = 0;
long lGetHessianRowCounter = 0;
long lRamHitCount = 0;
long lSSDHitCount = 0;
float_point* CSMOSolver::GetHessianRow(const int &nNumofTrainingSamples, const int &nPosofRow)
{
/*	timespec time1, time2;
	clock_gettime(CLOCK_REALTIME, &time1);
	lGetHessianRowCounter++;
*/
	float_point *pfDevHessianRow = NULL;
	//get 1st row
	int nCacheLocation = -1;
	bool bIsCacheFull = false;
	bool bIsInCache = m_pGPUCache->GetDataFromCache(nPosofRow, nCacheLocation, bIsCacheFull);

	if(bIsInCache == false)	//cache missed
	{
		if(bIsCacheFull == true)
			m_pGPUCache->ReplaceExpired(nPosofRow, nCacheLocation, m_pfDevGValue);
		//convert current training position to the position in Hessian matrix
		int nPosofRowAtHessian = nPosofRow;
		bool bMapIndex = MapIndexToHessian(nPosofRowAtHessian);
		//assert(bMapIndex == true);
		//if the hessian row is in host memory
		if(m_pHessianReader->m_nNumofCachedHessianRow > nPosofRowAtHessian)
		{
			lRamHitCount++;
			int nSizeofFirstPart = 0;
			if(m_pHessianReader->m_nRowStartPos1 != -1)
			{
				nSizeofFirstPart = m_pHessianReader->m_nRowEndPos1 - m_pHessianReader->m_nRowStartPos1 + 1;//the size of first part (include the last element of the part)
				long long nIndexofFirstElement = (long long)nPosofRowAtHessian * m_pHessianReader->m_nTotalNumofInstance + m_pHessianReader->m_nRowStartPos1;
				memcpy(m_pfHessianRow, m_pHessianReader->m_pfHessianRowsInHostMem + nIndexofFirstElement, nSizeofFirstPart * sizeof(float_point));
			}
			if(m_pHessianReader->m_nRowStartPos2 != -1)
			{
				int nSizeofSecondPart = m_pHessianReader->m_nRowEndPos2 - m_pHessianReader->m_nRowStartPos2 + 1;
				long long nIndexofFirstElement = (long long)nPosofRowAtHessian * m_pHessianReader->m_nTotalNumofInstance + m_pHessianReader->m_nRowStartPos2;
				memcpy(m_pfHessianRow + nSizeofFirstPart, m_pHessianReader->m_pfHessianRowsInHostMem + nIndexofFirstElement,
					   nSizeofSecondPart * sizeof(float_point));
			}
		}
		else//the hessian row is in SSD
		{
			lSSDHitCount++;
			m_pHessianReader->ReadHessianRow(m_pFile, nPosofRowAtHessian, m_pfHessianRow);
		}
		/*if(nPosofRow == 128)
		{
			for(int i = 0; i < 1800; i++)
			{
				cout << m_pfHessianRow[i] << "\t";
				if(i % 100 == 0 && i != 0)
				{
					cout << endl;
				}
			}
			exit(0);
		}*/

		checkCudaErrors(cudaMemcpyAsync(m_pfDevHessianMatrixCache + nCacheLocation * m_nNumofElementEachRowInCache, m_pfHessianRow,
							  	  	sizeof(float_point) * nNumofTrainingSamples, cudaMemcpyHostToDevice, m_stream1_Hessian_row));
	}
	pfDevHessianRow = m_pfDevHessianMatrixCache + nCacheLocation * m_nNumofElementEachRowInCache;
/*	clock_gettime(CLOCK_REALTIME, &time2);

	long lTemp = ((time2.tv_sec - time1.tv_sec) * 1e9 + (time2.tv_nsec - time1.tv_nsec));
	//if(lTemp > 0)
	{
		lGetHessianRowTime += lTemp;
	}*/
	return pfDevHessianRow;
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
	cudaDeviceSynchronize();
	//####### timer
	timeval tSelect1S, tSelect1E;
	gettimeofday(&tSelect1S, NULL);

/*	cudaProfilerStart();
*/	//block level reducer
	GetBlockMinYiGValue<<<dimGridThinThread, BLOCK_SIZE>>>(pfDevYiFValue, pfDevAlpha, pnDevLabel, gfPCost,
												 nNumofTrainingSamples, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey, m_nNumofBlock, pfDevYiFValue, NULL, m_pfDevBuffer);


/*	//when the number of blocks is larger than 65535, compute the number of grid
	int nNumofBlock = Ceil(nNumofTrainingSamples, (BLOCK_SIZE * TASK_OF_THREAD));
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
	cudaMemcpy(m_pfHostBuffer, m_pfDevBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
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
	cudaStreamSynchronize(m_stream1_Hessian_row);

	//cudaProfilerStart();
	//get block level min (-b_ij*b_ij/a_ij)
/**/	GetBlockMinLowValue<<<dimGridThinThread, BLOCK_SIZE>>>
					   (pfDevYiFValue, pfDevAlpha, pnDevLabel, gfNCost, nNumofTrainingSamples, m_pfDevDiagHessian,
						m_pfDevHessianSampleRow1, m_fUpValue, fUpSelfKernelValue, m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
						m_pfDevBlockMinYiFValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE>>>
				(m_pfDevBlockMin, m_pnDevBlockMinGlobalKey,
				 m_nNumofBlock, pfDevYiFValue, m_pfDevHessianSampleRow1, m_pfDevBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(m_pfDevBlockMinYiFValue, m_nNumofBlock, m_pfDevBuffer);

/*	GetBigBlockMinLowValue<<<temp, BLOCK_SIZE>>>
					   (pfDevYiFValue, pfDevAlpha, pnDevLabel, gfNCost, nNumofTrainingSamples, m_pfDevDiagHessian,
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
	cudaThreadSynchronize();
	//copy result back to host
	cudaMemcpy(m_pfHostBuffer, m_pfDevBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
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

	//assert(nNumofTrainingSamples > 0 && m_nIndexofSampleOne >= 0 && m_nIndexofSampleTwo >= 0 &&
	///	   m_nIndexofSampleOne < nNumofTrainingSamples && m_nIndexofSampleTwo < nNumofTrainingSamples);

	//get YiGValue for sample one and two
	float_point fAlpha2 = 0;
	float_point fYiFValue2 = 0;
	fAlpha2 = m_pfAlpha[m_nIndexofSampleTwo];
	fYiFValue2 = fMinLowValue;

	//get alpha values of sample
	float_point fAlpha1 = 0;
	float_point fYiFValue1 = 0;
	fAlpha1 = m_pfAlpha[m_nIndexofSampleOne];
	fYiFValue1 = fMinValue;

	//Get K(x_up, x_up), and K(x_low, x_low)
	float_point fDiag1 = 0, fDiag2 = 0;
	fDiag1 = m_pfDiagHessian[m_nIndexofSampleOne];
	fDiag2 = m_pfDiagHessian[m_nIndexofSampleTwo];

	//get labels of sample one and two
	int nLabel1 = 0, nLabel2 = 0;
	nLabel1 = m_pnLabel[m_nIndexofSampleOne];
	nLabel2 = m_pnLabel[m_nIndexofSampleTwo];

	//compute eta
	float_point eta = fDiag1 + fDiag2 - 2 * fKernelValue;
	if (eta <= 0)
		eta = TAU;

	float_point fCost1, fCost2;
	fCost1 = Get_C(nLabel1);
	fCost2 = Get_C(nLabel2);

	//keep old yi*alphas
	fY1AlphaDiff = nLabel1 * fAlpha1;
	fY2AlphaDiff = nLabel2 * fAlpha2;

	//get new alpha values
	int nSign = nLabel2 * nLabel1;
	if(nSign < 0)
	{
		float_point fDelta = (-nLabel1 * fYiFValue1 - nLabel2 * fYiFValue2) / eta; //(-fYiFValue1 - fYiFValue2) / eta;
		float_point fAlphaDiff = fAlpha1 - fAlpha2;
		fAlpha1 += fDelta;
		fAlpha2 += fDelta;

		if (fAlphaDiff > 0)
		{
			if (fAlpha2 < 0)
			{
				fAlpha2 = 0;
				fAlpha1 = fAlphaDiff;
			}
		}
		else
		{
			if (fAlpha1 < 0)
			{
				fAlpha1 = 0;
				fAlpha2 = -fAlphaDiff;
			}
		}

		if (fAlphaDiff > fCost1 - fCost2)
		{
			if (fAlpha1 > fCost1)
			{
				fAlpha1 = fCost1;
				fAlpha2 = fCost1 - fAlphaDiff;
			}
		}
		else
		{
			if (fAlpha2 > fCost2)
			{
				fAlpha2 = fCost2;
				fAlpha1 = fCost2 + fAlphaDiff;
			}
		}
	} //end if nSign < 0
	else
	{
		float_point fDelta = (nLabel1 * fYiFValue1 - nLabel2 * fYiFValue2) / eta;
		float_point fSum = fAlpha1 + fAlpha2;
		fAlpha1 -= fDelta;
		fAlpha2 += fDelta;

		if (fSum > fCost1)
		{
			if (fAlpha1 > fCost1)
			{
				fAlpha1 = fCost1;
				fAlpha2 = fSum - fCost1;
			}
		}
		else
		{
			if (fAlpha2 < 0)
			{
				fAlpha2 = 0;
				fAlpha1 = fSum;
			}
		}
		if (fSum > fCost2)
		{
			if (fAlpha2 > fCost2)
			{
				fAlpha2 = fCost2;
				fAlpha1 = fSum - fCost2;
			}
		}
		else
		{
			if (fAlpha1 < 0)
			{
				fAlpha1 = 0;
				fAlpha2 = fSum;
			}
		}
	}//end get new alpha values

//	cout << m_nIndexofSampleOne << " v.s. " << m_nIndexofSampleTwo << endl;
/*
	cout << "index1=" << m_nIndexofSampleOne << "; alpha1=" << m_pfAlpha[m_nIndexofSampleOne] <<
			"; index2=" << m_nIndexofSampleTwo << "; alpha2=" << m_pfAlpha[m_nIndexofSampleTwo] <<
			"; alpha1new=" << fAlpha1 << "; alpha2new=" << fAlpha2 << endl;
	sleep(1);
	*/

	m_pfAlpha[m_nIndexofSampleOne] = fAlpha1;
	m_pfAlpha[m_nIndexofSampleTwo] = fAlpha2;
	//update I_up and I_low group
	//m_pCache->UpdateGroup(fAlpha1, nLabel1, fCost1, m_nIndexofSampleOne);
	//m_pCache->UpdateGroup(fAlpha2, nLabel2, fCost2, m_nIndexofSampleTwo);
	m_pGPUCache->UnlockCacheEntry(m_nIndexofSampleOne);

	//get alpha difference
	fY1AlphaDiff = nLabel1 * fAlpha1 - fY1AlphaDiff; //(alpha1' - alpha1) * y1
	fY2AlphaDiff = nLabel2 * fAlpha2 - fY2AlphaDiff;

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
	cudaMemcpy(m_pfDevBuffer, m_pfHostBuffer, sizeof(float_point) * 4, cudaMemcpyHostToDevice);
	UpdateYiFValueKernel<<<dimGridThinThread, BLOCK_SIZE>>>(pfDevAlpha, m_pfDevBuffer, pfDevYiFValue,
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
