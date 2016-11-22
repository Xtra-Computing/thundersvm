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

#include "trainingFunction.h"
#include "cvFunction.h"
#include "commandLineParser.h"
#include "../svm-shared/initCuda.h"
#include "svmModel.h"
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	char fileName[1024];
	char savedFileName[1024];
	Parser parser;
	parser.ParseLine(argc, argv, fileName, savedFileName);

    CUcontext context;
	if(!InitCUDA(context,'G'))
	{
		return 0;
	}

    WorkParam::devContext = context;
	printf("CUDA initialized.\n");

	if(parser.task_type == 1)
	{
		//perform cross validation*/
		cout << "performing cross-validation" << endl;
		crossValidation(parser.param, fileName);
	}
    else if(parser.task_type == 0)
    {
		//perform svm training
		cout << "performing training" << endl;
		SvmModel model;
		trainSVM(parser.param, fileName, parser.nNumofFeature, model);
    }
    else if(parser.task_type == 2)
    {
 		//perform svm evaluation 
		cout << "performing evaluation" << endl;
		SvmModel model;
		trainSVM(parser.param, fileName, parser.nNumofFeature, model);
        evaluateSVMClassifier(model, fileName, parser.nNumofFeature);
       
    }

    return 0;
}
