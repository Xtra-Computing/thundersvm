
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
	InitSolver(nNumofTrainingSamples);

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
	checkCudaErrors(cudaFree(devHessianDiag));

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
	DeInitSolver();

	cudaFreeHost(m_pfHessianRow);
	delete[] m_pnLabel;
	delete[] hessianDiag;
	delete[] m_pfGValue;

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
long readRowTime = 0;
long cacheMissMemcpyTime = 0;
long lGetHessianRowCounter = 0;
long cacheMissCount = 0;
long lRamHitCount = 0;
long lSSDHitCount = 0;
float_point* CSMOSolver::GetHessianRow(const int &nNumofInstance, const int &nPosofRow)
{
	timespec time1, time2, time3, time4, time5;
	clock_gettime(CLOCK_REALTIME, &time1);
	lGetHessianRowCounter++;
    /*printf("get row %d label %d\n",problem->originalIndex[nPosofRow], problem->originalLabel[nPosofRow]);*/
//	assert(nNumofInstance >= nPosofRow);

	float_point *pfDevHessianRow = NULL;
	//get 1st row
	int nCacheLocation = -1;
	bool bIsCacheFull = false;
	bool bIsInCache = m_pGPUCache->GetDataFromCache(nPosofRow, nCacheLocation, bIsCacheFull);

	long long lCachePosStart = (long long)nCacheLocation * m_lNumofElementEachRowInCache;

	if(bIsInCache == false)
	{//cache missed
		clock_gettime(CLOCK_REALTIME, &time3);
		if(bIsCacheFull == true)
			m_pGPUCache->ReplaceExpired(nPosofRow, nCacheLocation, NULL);
		//convert current training position to the position in Hessian matrix
		int nPosofRowAtHessian = nPosofRow;
		bool bMapIndex = MapIndexToHessian(nPosofRowAtHessian);
		assert(bMapIndex == true);

//		memset(m_pfHessianRow, 0, sizeof(float_point) * nNumofInstance);
		m_pHessianReader->ReadRow(nPosofRowAtHessian, m_pfHessianRow);

		clock_gettime(CLOCK_REALTIME, &time4);

//		cout << nCacheLocation << "; cache is full=" << bIsCacheFull << endl;
		lCachePosStart = (long long)nCacheLocation * m_lNumofElementEachRowInCache;
		//checkCudaErrors(cudaMemcpyAsync(m_pfDevHessianMatrixCache + lCachePosStart, m_pfHessianRow,
		//					  	  		sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice, m_stream1_Hessian_row));
		checkCudaErrors(cudaMemcpy((m_pfDevHessianMatrixCache + lCachePosStart), m_pfHessianRow, sizeof(float_point) * nNumofInstance, cudaMemcpyHostToDevice));

		cacheMissCount++;
		clock_gettime(CLOCK_REALTIME, &time5);
		long lTemp = ((time5.tv_sec - time3.tv_sec) * 1e9 + (time5.tv_nsec - time3.tv_nsec));
		readRowTime += lTemp;
		lTemp = ((time5.tv_sec - time4.tv_sec) * 1e9 + (time5.tv_nsec - time4.tv_nsec));
        cacheMissMemcpyTime += lTemp;
	}

	pfDevHessianRow = m_pfDevHessianMatrixCache + lCachePosStart;
	clock_gettime(CLOCK_REALTIME, &time2);
	long lTemp = ((time2.tv_sec - time1.tv_sec) * 1e9 + (time2.tv_nsec - time1.tv_nsec));
	if(lTemp > 0)
	{
		lGetHessianRowTime += lTemp;
	}

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
	fAlpha2 = alpha[IdofInstanceTwo];
	fYiFValue2 = fMinLowValue;

	//get alpha values of sample
	float_point fAlpha1 = 0;
	float_point fYiFValue1 = 0;
	fAlpha1 = alpha[IdofInstanceOne];
	fYiFValue1 = fMinValue;

	//Get K(x_up, x_up), and K(x_low, x_low)
	float_point fDiag1 = 0, fDiag2 = 0;
	fDiag1 = hessianDiag[nHessianRowOneInMatrix];
	fDiag2 = hessianDiag[nHessianRowTwoInMatrix];

	//get labels of sample one and two
	int nLabel1 = 0, nLabel2 = 0;
	nLabel1 = m_pnLabel[IdofInstanceOne];
	nLabel2 = m_pnLabel[IdofInstanceTwo];

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

	alpha[IdofInstanceOne] = fAlpha1;
    alpha[IdofInstanceTwo] = fAlpha2;

	//get alpha difference
	fY1AlphaDiff = nLabel1 * fAlpha1 - fY1AlphaDiff; //(alpha1' - alpha1) * y1
	fY2AlphaDiff = nLabel2 * fAlpha2 - fY2AlphaDiff;
}

