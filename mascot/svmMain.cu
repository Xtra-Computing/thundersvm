/*
 * svmMain.cu
 *
 *  Created on: May 21, 2012
 *      Author: Zeyi Wen
 */

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "cvFunction.h"
#include "commandLineParser.h"
#include "../SharedUtility/initCuda.h"
#include "svmModel.h"
#include "trainClassifier.h"
#include "classifierEvaluater.h"
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	char fileName[1024];
	char savedFileName[1024];
	Parser parser;
	parser.ParseLine(argc, argv, fileName, savedFileName);

    CUcontext context;
	if(!InitCUDA(context, 'G'))
		return 0;
	printf("CUDA initialized.\n");

    if(parser.task_type == 0){
		//perform svm training
		cout << "performing training" << endl;
		SvmModel model;
		trainSVM(parser.param, fileName, parser.nNumofFeature, model, parser.compute_training_error);
    }else if(parser.task_type == 1){
		//perform cross validation*/
		cout << "performing cross-validation" << endl;
		crossValidation(parser.param, fileName);
	}else if(parser.task_type == 2){
 		//perform svm evaluation 
		cout << "performing evaluation" << endl;
		SvmModel model;
        cout << "start training..." << endl;
		trainSVM(parser.param, fileName, parser.nNumofFeature, model, parser.compute_training_error);
        cout << "start evaluation..." << endl;
        evaluateSVMClassifier(model, strcat(fileName, ".t"), parser.nNumofFeature);
    }else if(parser.task_type == 4){
    	//perform selecting best C
    	cout << "perform C selection" << endl;
    	vector<float_point> vC;
    	for(int i = 0; i < 3; i++){
			SvmModel model;
			model.vC = vC;
			cout << "start training..." << endl;
			trainSVM(parser.param, fileName, parser.nNumofFeature, model, parser.compute_training_error);
			cout << "start evaluation..." << endl;
			evaluateSVMClassifier(model, strcat(fileName, ".t"), parser.nNumofFeature);
			vC = ClassifierEvaluater::updateC(model.vC);
    	}
    }
	else if(parser.task_type == 3){
    	cout << "performing grid search" << endl;
    	Grid paramGrid;
    	paramGrid.vfC.push_back(parser.param.C);
    	paramGrid.vfGamma.push_back(parser.param.gamma);
    	gridSearch(paramGrid, fileName);
    }else{
    	cout << "unknown task type" << endl;
    }

    return 0;
}
