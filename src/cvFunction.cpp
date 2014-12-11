/*
 * cvFunction.cpp
 *
 *  Created on: 09/12/2014
 *      Author: Zeyi Wen
 */

#include<iostream>
#include<cassert>
#include <stdio.h>
using std::cout;
using std::endl;

#include "gpu_global_utility.h"
#include "HessianIO/seqHessianIO.h"
#include "HessianIO/parHessianIO.h"
//#include "hessianIO.h"
#include "kernelCalculater/kernelCalculater.h"
#include "svmTrainer.h"
#include "svmPredictor.h"
#include "modelSelector.h"
#include "smoSolver.h"
#include "Cache/cache.h"
#include "fileOps.h"
#include "DataIOOps/DataIO.h"
//device function for CPairSelector

#include "classificationKernel.h"
#include "cvFunction.h"

void crossValidation(SVMParam &param, string strTrainingFileName, int nNumofFeature)
{
	gnNumofThread = 16;
	lIO_timer = 0;

	//initialize grid
	grid SGrid;
	SGrid.nNumofGamma = 1;
	SGrid.nNumofC = 1;

	SGrid.pfCost = new float_point[SGrid.nNumofC];
	SGrid.pfCost[0] = param.C;
	SGrid.pfGamma = new float_point[SGrid.nNumofGamma];
	SGrid.pfGamma[0] = param.gamma;
	for(int i = 1; i < SGrid.nNumofGamma; i++)
	{
		SGrid.pfGamma[i] = SGrid.pfGamma[i - 1] * 4;
	}
	for(int i = 1; i < SGrid.nNumofC; i++)
	{
		SGrid.pfCost[i] = SGrid.pfCost[i - 1] * 4;
	}

	CDataIOOps rawDataRead;
	vector<vector<float_point> > v_vDocVector;
	vector<int> v_nLabel;
	rawDataRead.ReadFromFile(strTrainingFileName, nNumofFeature, v_vDocVector, v_nLabel);
//	cout << "data read " << v_vDocVector.size() << " : " << v_nLabel.size() << endl;

	CModelSelector modelSelector;

	timeval t1, t2;
	float_point elapsedTime;
	gettimeofday(&t1, NULL);
	modelSelector.GridSearch(SGrid, v_vDocVector, v_nLabel);
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
	//cout << elapsedTime << " ms.\n";
}
