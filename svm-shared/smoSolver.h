/**
 * workingSetSelector.h
 * This file contains kernel functions for working set selection
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 **/

#ifndef WORKINGSETSELECTOR_H_
#define WORKINGSETSELECTOR_H_

#include "float.h"
#include <string>
#include <fstream>

//Self define header files
#include "Cache/cache.h"
#include "smoGPUHelper.h"
#include "HessianIO/deviceHessian.h"
#include "../mascot/svmProblem.h"
#include "baseSMO.h"

using std::string;
using std::ifstream;

extern long lGetHessianRowTime;
extern long lGetHessianRowCounter;
extern long lRamHitCount;
extern long lSSDHitCount;
class CSMOSolver: public BaseSMO
{
public:
    SvmProblem *problem;
	real m_fLowValue;

	CCache *m_pGPUCache;
	BaseHessian *m_pHessianReader;

	int m_nStart1, m_nEnd1, m_nStart2, m_nEnd2;
	//members for cpu, gpu communication
	real *m_pfHessianRow;//this member is for re-using memory (avoid allocating memory)
	long long m_lNumofElementEachRowInCache;
	real *m_pfGValue;
	int *m_pnLabel;

	//for Hessian Matrix
	real *m_pfDevHessianMatrixCache;

public:
	CSMOSolver(BaseHessian *pHessianOps, CCache *pCache)
	{
		IdofInstanceOne = IdofInstanceTwo = -1;
		upValue = -1;
		devBlockMin = NULL;
		devBlockMinYiGValue = NULL;
		devBlockMinGlobalKey = NULL;
		devMinValue = NULL;
		devMinKey = NULL;

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
	}
	~CSMOSolver(){}

    virtual real *ObtainRow(int numTrainingInstance)
    {
    	return GetHessianRow(numTrainingInstance, IdofInstanceOne);
    }

	void SetCacheStrategy(CCache *pCache){m_pGPUCache = pCache;}
	void SetCacheSize(int nCacheSize){m_pGPUCache->SetCacheSize(nCacheSize);}

	/******* preparation for start SMO solver ************/
	bool SMOSolverPreparation(const int&);
	bool SMOSolverEnd();

	bool InitCache(const int &nCacheSize, const int &nNumofTrainingSamples);
	bool ReadToCache(const int &nStartRow, const int &nEndRow, const int &nNumofTrainingSamples, real *pfDevHessianCacheEndPos);
	bool CleanCache();

	//set involve data in Hessian matrix
	bool SetInvolveData(int nStart1, int nEnd1, int nStart2, int nEnd2);

	bool MapIndexToHessian(int &nIndex);
	real *GetHessianRow(const int &nNumofInstance, const int &nPosofRow);

	real Get_C(int nLabel)
	{
		return (nLabel > 0)? gfPCost : gfNCost;
	}

	int Iterate(real *pfDevYiFValue, real *pfDevAlpha, int *npDevLabel, const int &nNumofTrainingSamples);
	int IterateAdv(real*, int*, const int &nNumofTrainingSamples);

	/***************** functions for testing purposes *************************/
private:
	real *m_pfRow40;
	int m_nCcounter;
	void StoreRow(real *pfDevRow, int nLen);
	void PrintTenGPUHessianRow(real *pfDevRow, int nLen);
	void PrintGPUHessianRow(real *pfDevRow, int nLen);
	int CompareTwoGPURow(real *pfDevRow1, real *pfDevRow2, int nLen);
};

#endif /* WORKINGSETSELECTOR_H_ */
