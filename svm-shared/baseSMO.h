/*
 * baseSMO.h
 *	@brief: some sharable functions of smo solver
 *  Created on: 24 Dec 2016
 *      Author: Zeyi Wen
 */

#ifndef SVM_SHARED_BASESMO_H_
#define SVM_SHARED_BASESMO_H_

#include "host_constant.h"

class BaseSMO
{
public:
    BaseSMO(){}
	virtual ~BaseSMO(){}
	void SelectFirst(int numTrainingInstance, float_point CforPositive);
	void SelectSecond(int numTrainingInstance, float_point CforNegative);

	void UpdateYiGValue(int numTrainingInstance, float_point fY1AlphaDiff, float_point fY2AlphaDiff);

protected:
	virtual float_point *ObtainRow(int numTrainingInstance) = 0;

protected:
	vector<float_point> alpha;
    float_point upValue;
    float_point lowValue;
    float_point *devBuffer;
    float_point *hostBuffer;
    int IdofInstanceOne;
    int IdofInstanceTwo;

    float_point *devHessianDiag;
    float_point *hessianDiag;			//diagonal of the hessian matrix
    float_point *devHessianInstanceRow1;//kernel values of the first instance
    float_point *devHessianInstanceRow2;	//kernel values of the second instance

    float_point *devAlpha;
    float_point *devYiGValue;
    int *devLabel;

    float_point *devBlockMin;			//for reduction in min/max search
    int *devBlockMinGlobalKey;			//for reduction in min/max search
    float_point *devBlockMinYiGValue;

    int numOfBlock;
    dim3 gridSize;
};



#endif /* SVM_SHARED_BASESMO_H_ */
