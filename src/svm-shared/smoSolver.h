/**
 * workingSetSelector.h
 * This file contains kernel functions for working set selection
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef WORKINGSETSELECTOR_H_
#define WORKINGSETSELECTOR_H_

#include "float.h"
#include <string>
#include <fstream>

//Self define header files
#include "Cache/cache.h"
#include "smoGPUHelper.h"
#include "HessianIO/hessianIO.h"
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

using std::string;
using std::ifstream;

extern long lGetHessianRowTime;
extern long lGetHessianRowCounter;
extern long lRamHitCount;
extern long lSSDHitCount;
class CSMOSolver
{
public:
	int m_nIndexofSampleOne;
	int m_nIndexofSampleTwo;
	float_point m_fUpValue;
	float_point m_fLowValue;
	FILE *m_pFile;
	boost::interprocess::file_mapping *pFm;
	boost::interprocess::mapped_region *pRegion;

	CCache *m_pGPUCache;
	CHessianIOOps *m_pHessianReader;

	int m_nStart1, m_nEnd1, m_nStart2, m_nEnd2;
	//members for cpu, gpu communication
	float_point *m_pfHessianRow;//this member is for re-using memory (avoid allocating memory)
	long long m_lNumofElementEachRowInCache;
	float_point *m_pfGValue;
	float_point *m_pfAlpha;
	int *m_pnLabel;
	float_point *m_pfDiagHessian;
	float_point *m_pfHostBuffer;

	//members for gpu
	float_point *m_pfDevBlockMin;
	float_point *m_pfDevBlockMinYiFValue; //for getting maximum low value
	int *m_pnDevBlockMinGlobalKey;
	float_point *m_pfDevMinValue;
	int *m_pnDevMinKey;
	float_point *m_pfDevGValue;
	float_point *m_pfDevBuffer;

	//kernel launching parameters
	dim3 dimGridThinThread;
	int m_nNumofBlock;
	cudaStream_t m_stream1_Hessian_row;
	cudaStream_t m_stream_memcpy;

	//for Hessian Matrix
	float_point *m_pfDevDiagHessian;
	float_point *m_pfDevHessianMatrixCache;
	float_point *m_pfDevHessianSampleRow1;
	float_point *m_pfDevHessianSampleRow2;

public:
	CSMOSolver(CHessianIOOps *pHessianOps, CCache *pCache)
	{
		m_nIndexofSampleOne = m_nIndexofSampleTwo = -1;
		m_fUpValue = -1;
		m_pfDevBlockMin = NULL;
		m_pfDevBlockMinYiFValue = NULL;
		m_pnDevBlockMinGlobalKey = NULL;
		m_pfDevMinValue = NULL;
		m_pnDevMinKey = NULL;

		//for n-fold-cross validation
		m_nStart1 = -1;
		m_nEnd1 = -1;
		m_nStart2 = -1;
		m_nEnd2 = -1;

		//objects initialization
		m_pHessianReader = pHessianOps;
		m_pGPUCache = pCache;
		/*if(m_pFile != NULL)
		{
			cout << "file closed before opening a new file" << endl;
			fclose(m_pFile);
		}*/
		//create a file when it doesn't exist yet
//commented out if you want to reuse the precomputed matrix
//5/5 (1-4 are in hessianIO.cu file)
/*		m_pFile = fopen(HESSIAN_FILE, "w");
		if(m_pFile == NULL)
		{
			cerr << "failed to open: \"" << HESSIAN_FILE << "\" as Hessian file" << endl;
			exit(0);
		}
		fclose(m_pFile);
*/
		m_pFile = fopen(HESSIAN_FILE, "rb");
		if(m_pFile == NULL)
		{
			cerr << "failed to open: \"" << HESSIAN_FILE << "\" as Hessian file" << endl;
			exit(0);
		}
		//pFm = new boost::interprocess::file_mapping(HESSIAN_FILE, boost::interprocess::read_only);
		//pRegion = new boost::interprocess::mapped_region(*pFm, boost::interprocess::read_only);
	}
	~CSMOSolver()
	{
		fclose(m_pFile);
	}

	void SetCacheStrategy(CCache *pCache){m_pGPUCache = pCache;}
	void SetCacheSize(int nCacheSize){m_pGPUCache->SetCacheSize(nCacheSize);}

	/******* preparation for start SMO solver ************/
	bool SMOSolverPreparation(const int&);
	bool SMOSolverEnd();

	bool InitCache(const int &nCacheSize, const int &nNumofTrainingSamples);
	bool ReadToCache(const int &nStartRow, const int &nEndRow, const int &nNumofTrainingSamples, float_point *pfDevHessianCacheEndPos);
	bool CleanCache();

	//set involve data in Hessian matrix
	bool SetInvolveData(int nStart1, int nEnd1, int nStart2, int nEnd2);

	bool MapIndexToHessian(int &nIndex);
	float_point *GetHessianRow(const int &nNumofInstance, const int &nPosofRow);

	float_point Get_C(int nLabel)
	{
		return (nLabel > 0)? gfPCost : gfNCost;
	}

	void UpdateTwoWeight(float_point fMinLowValue, float_point fMinValue,
			 	 	 	 int nHessianRowOneInMatrix, int nHessianRowTwoInMatrix,
			 	 	 	 float_point fUpSelfKernelValue, float_point &fY1AlphaDiff, float_point &fY2AlphaDiff);

	int Iterate(float_point *pfDevYiFValue, float_point *pfDevAlpha, int *npDevLabel, const int &nNumofTrainingSamples);
	int IterateAdv(float_point*, int*, const int &nNumofTrainingSamples);

	/***************** functions for testing purposes *************************/
private:
	float_point *m_pfRow40;
	int m_nCcounter;
	void StoreRow(float_point *pfDevRow, int nLen);
	void PrintTenGPUHessianRow(float_point *pfDevRow, int nLen);
	void PrintGPUHessianRow(float_point *pfDevRow, int nLen);
	int CompareTwoGPURow(float_point *pfDevRow1, float_point *pfDevRow2, int nLen);
};

#endif /* WORKINGSETSELECTOR_H_ */
