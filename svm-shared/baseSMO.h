/*
 * baseSMO.h
 *	@brief: some sharable functions of smo solver
 *  Created on: 24 Dec 2016
 *      Author: Zeyi Wen
 */

#ifndef SVM_SHARED_BASESMO_H_
#define SVM_SHARED_BASESMO_H_

#include <vector>
#include "../SharedUtility/DataType.h"
using std::vector;

class BaseSMO
{
public:
    BaseSMO(){}
	virtual ~BaseSMO(){}

	void InitSolver(int nNumofTrainingIns);
	void DeInitSolver();

	void SelectFirst(int numTrainingInstance, real CforPositive);
	void SelectSecond(int numTrainingInstance, real CforNegative);
	void UpdateTwoWeight(real fMinLowValue, real fMinValue, int nHessianRowOneInMatrix,
	                                     int nHessianRowTwoInMatrix, real fKernelValue, real &fY1AlphaDiff,
	                                     real &fY2AlphaDiff, const int *label, real C);
	void UpdateYiGValue(int numTrainingInstance, real fY1AlphaDiff, real fY2AlphaDiff);

protected:
	virtual real *ObtainRow(int numTrainingInstance) = 0;

public:
    real upValue;
	vector<real> alpha;
protected:
    real lowValue;
    real *devBuffer;
    real *hostBuffer;
    int IdofInstanceOne;
    int IdofInstanceTwo;

    real *devHessianDiag;
    real *hessianDiag;			//diagonal of the hessian matrix
    real *devHessianInstanceRow1;//kernel values of the first instance
    real *devHessianInstanceRow2;	//kernel values of the second instance

    real *devAlpha;
    real *devYiGValue;
    int *devLabel;

    real *devBlockMin;			//for reduction in min/max search
    int *devBlockMinGlobalKey;			//for reduction in min/max search
    real *devBlockMinYiGValue;

	real *devMinValue;			//store the min/max value
	int *devMinKey;						//store the min/max key

    int numOfBlock;
    dim3 gridSize;
	void configureCudaKernel(int numOfTrainingInstance);
};



#endif /* SVM_SHARED_BASESMO_H_ */
