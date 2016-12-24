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
	float_point m_fLowValue;

	CCache *m_pGPUCache;
	BaseHessian *m_pHessianReader;

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

	//for Hessian Matrix
	float_point *m_pfDevDiagHessian;
	float_point *m_pfDevHessianMatrixCache;

public:
	CSMOSolver(BaseHessian *pHessianOps, CCache *pCache)
	{
		IdofInstanceOne = IdofInstanceTwo = -1;
		upValue = -1;
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
	}
	~CSMOSolver(){}

    virtual float_point *ObtainRow(int numTrainingInstance)
    {
    	devHessianInstanceRow1 = GetHessianRow(numTrainingInstance, IdofInstanceOne);

        //lock cached entry for the sample one, in case it is replaced by sample two
        m_pGPUCache->LockCacheEntry(IdofInstanceOne);
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
