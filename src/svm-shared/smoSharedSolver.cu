
#include "smoSolver.h"
#include "smoGPUHelper.h"
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
//#include <cutil.h>
#include <helper_cuda.h>
#include <time.h>
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

	m_pfHostBuffer = new float_point[5];
	checkCudaErrors(cudaMalloc((void**)&m_pfDevBuffer, sizeof(float_point) * 5));//only need 4 float_points

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
		cerr << "CUDA error occurs at freeing GPU memory for SMOSolver" << endl;
		bReturn = false;
	}

	cudaFreeHost(m_pfHessianRow);
	delete[] m_pnLabel;
	delete[] m_pfDiagHessian;
	delete[] m_pfGValue;
	delete[] m_pfAlpha;
	delete[] m_pfHostBuffer;

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
float_point* CSMOSolver::GetHessianRow(const int &nNumofInstance, const int &nPosofRow)
{
/*	timespec time1, time2;
	clock_gettime(CLOCK_REALTIME, &time1);
*/	lGetHessianRowCounter++;
	assert(nNumofInstance >= nPosofRow);

	float_point *pfDevHessianRow = NULL;
	//get 1st row
	int nCacheLocation = -1;
	bool bIsCacheFull = false;
	bool bIsInCache = m_pGPUCache->GetDataFromCache(nPosofRow, nCacheLocation, bIsCacheFull);

	long long lCachePosStart = (long long)nCacheLocation * m_lNumofElementEachRowInCache;

	/*if(m_nIndexofSampleOne == 24530 && m_nIndexofSampleTwo == 17958)
	{
		cout << "is in cache=" << bIsInCache << "; location=" << nCacheLocation << endl;
		cout << "is cache full=" << bIsCacheFull << "; pos of row=" << nPosofRow << endl;
		cout << lCachePosStart << endl;
		cout << "before copying: ";
		PrintTenGPUHessianRow(m_pfDevHessianMatrixCache + 40 * m_lNumofElementEachRowInCache, nNumofInstance);
	}*/
	if(bIsInCache == false)
	{//cache missed
		if(bIsCacheFull == true)
			m_pGPUCache->ReplaceExpired(nPosofRow, nCacheLocation, m_pfDevGValue);
		//convert current training position to the position in Hessian matrix
		int nPosofRowAtHessian = nPosofRow;
		bool bMapIndex = MapIndexToHessian(nPosofRowAtHessian);
//		if(m_nIndexofSampleOne == 24530 && m_nIndexofSampleTwo == 17958)
//			cout << "CPU cached rows " << m_pHessianReader->m_nNumofCachedHessianRow << "; pos at hessian is " << nPosofRowAtHessian << endl;
		assert(bMapIndex == true);

		memset(m_pfHessianRow, 0, sizeof(float_point) * nNumofInstance);
		//if the hessian row is in host memory
		if(m_pHessianReader->m_nNumofCachedHessianRow > nPosofRowAtHessian)
		{
			lRamHitCount++;
			int nSizeofFirstPart = 0;
			if(m_pHessianReader->m_nRowStartPos1 != -1)
			{
				nSizeofFirstPart = m_pHessianReader->m_nRowEndPos1 - m_pHessianReader->m_nRowStartPos1 + 1;//the size of first part (include the last element of the part)
				long long nIndexofFirstElement = (long long)nPosofRowAtHessian * (m_pHessianReader->m_nTotalNumofInstance) + m_pHessianReader->m_nRowStartPos1;
				memcpy(m_pfHessianRow, m_pHessianReader->m_pfHessianRowsInHostMem + nIndexofFirstElement, nSizeofFirstPart * sizeof(float_point));
			}
			if(m_pHessianReader->m_nRowStartPos2 != -1)
			{
				int nSizeofSecondPart = m_pHessianReader->m_nRowEndPos2 - m_pHessianReader->m_nRowStartPos2 + 1;
				long long nIndexofFirstElement = (long long)nPosofRowAtHessian * (m_pHessianReader->m_nTotalNumofInstance) + m_pHessianReader->m_nRowStartPos2;
				memcpy(m_pfHessianRow + nSizeofFirstPart, m_pHessianReader->m_pfHessianRowsInHostMem + nIndexofFirstElement,
					   nSizeofSecondPart * sizeof(float_point));
			}
		}
		else//the hessian row is in SSD
		{
			lSSDHitCount++;
			m_pHessianReader->ReadHessianRow(m_pFile, nPosofRowAtHessian, m_pfHessianRow);
		}

//		cout << nCacheLocation << "; cache is full=" << bIsCacheFull << endl;
		lCachePosStart = (long long)nCacheLocation * m_lNumofElementEachRowInCache;
/*		if(m_nIndexofSampleOne == 24530 && m_nIndexofSampleTwo == 17958)
		{
			cout << "cache location " << nCacheLocation << "; copy to here " << lCachePosStart
				 <<"; changed location is " << 40 * m_lNumofElementEachRowInCache << endl;
			cout << "before copying: ";
			PrintTenGPUHessianRow(m_pfDevHessianMatrixCache + 40 * m_lNumofElementEachRowInCache, nNumofInstance);
		}
		if(cudaGetLastError() != cudaSuccess)
		{
			cerr << "cuda error after initCuda" << endl;
			exit(-1);
		}

		cudaDeviceSynchronize();*/
		//checkCudaErrors(cudaMemcpyAsync(m_pfDevHessianMatrixCache + lCachePosStart, m_pfHessianRow,
		//					  	  		sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice, m_stream1_Hessian_row));
		checkCudaErrors(cudaMemcpy((m_pfDevHessianMatrixCache + lCachePosStart), m_pfHessianRow, sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice));
		/*cudaDeviceSynchronize();
		if(m_nIndexofSampleOne == 24530 && m_nIndexofSampleTwo == 17958)
		{
			cout << "after  copying: ";
			PrintTenGPUHessianRow(m_pfDevHessianMatrixCache + 40 * m_lNumofElementEachRowInCache, nNumofInstance);
		}*/
	}

	/*if(m_nIndexofSampleOne == 24530 && m_nIndexofSampleTwo == 17958)
	{
		cout << lCachePosStart << endl;
		cout << "after  copying: ";
		PrintTenGPUHessianRow(m_pfDevHessianMatrixCache + 40 * m_lNumofElementEachRowInCache, nNumofInstance);
	}*/

//	cout << lCachePosStart << endl;

	pfDevHessianRow = m_pfDevHessianMatrixCache + lCachePosStart;
/*	clock_gettime(CLOCK_REALTIME, &time2);
	long lTemp = ((time2.tv_sec - time1.tv_sec) * 1e9 + (time2.tv_nsec - time1.tv_nsec));
	//if(lTemp > 0)
	{
		lGetHessianRowTime += lTemp;
	}*/
	return pfDevHessianRow;
}


void CSMOSolver::UpdateTwoWeight(float_point fMinLowValue, float_point fMinValue,
								 int nHessianRowOneInMatrix, int nHessianRowTwoInMatrix,
								 float_point fKernelValue,
								 float_point &fY1AlphaDiff, float_point &fY2AlphaDiff)
{
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
	fDiag1 = m_pfDiagHessian[nHessianRowOneInMatrix];
	fDiag2 = m_pfDiagHessian[nHessianRowTwoInMatrix];

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

	m_pfAlpha[m_nIndexofSampleOne] = fAlpha1;
	m_pfAlpha[m_nIndexofSampleTwo] = fAlpha2;

	//get alpha difference
	fY1AlphaDiff = nLabel1 * fAlpha1 - fY1AlphaDiff; //(alpha1' - alpha1) * y1
	fY2AlphaDiff = nLabel2 * fAlpha2 - fY2AlphaDiff;
}

/*
 * @brief: read a few Hessian rows to GPU cache
 * @param: nNumofTrainingSamples: the number of samples for training
 * @param: sizeOfEachRowInCache: the size of each row in cache. This value is usually larger than nNumofTrainingSamples because of memory alignment
 * @param: pfDevHessianCacheEndPos: GPU cache end position
 */
//////////// This function is not used anymore
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
		long long nSpaceForCache = nNumofRowsToRead * m_lNumofElementEachRowInCache;//compute CPU cache size via size of each row in GPU cache due to the memory alignment
		float_point *pfHessianMatrixCache = new float_point[nSpaceForCache];
		memset(pfHessianMatrixCache, 0, sizeof(float_point) * nSpaceForCache);

		//start row and end row position
		int nTempStartRow = nStartRow + nNumofRowsRead;
		int nTempEndRow = nTempStartRow + nNumofRowsToRead - 1;	//as the index of nTempEndRow will be read
		//read Hessian rows
		m_pHessianReader->AllocateBuffer(nTempEndRow - nTempStartRow + 1);
		bReadHessian = m_pHessianReader->ReadHessianRows(m_pFile, nTempStartRow, nTempEndRow, nNumofTrainingSamples, pfHessianMatrixCache, m_lNumofElementEachRowInCache);
		m_pHessianReader->ReleaseBuffer();
		if(bReadHessian == false)
		{
			bReturn = false;
			break;
		}
		//end of cache location on GPU
		float_point *pfDevCacheEndPos = pfDevHessianCacheEndPos + nNumofRowsRead * m_lNumofElementEachRowInCache;
		checkCudaErrors(cudaMemcpy(pfDevCacheEndPos, pfHessianMatrixCache, sizeof(float_point) * nSpaceForCache, cudaMemcpyHostToDevice));
		nNumofRowsRead += nNumofRowsToRead;
		nNumofRowsToRead = nNumofRowsToRead2;
		delete[] pfHessianMatrixCache;
	}while(nNumofRowsRead < nNumofRows);

	return bReturn;
}


