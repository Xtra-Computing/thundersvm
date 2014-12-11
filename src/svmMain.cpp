/*
 * svmMain.cu
 *
 *  Created on: May 21, 2012
 *      Author: Zeyi Wen
 */

#include<iostream>
#include<cassert>
#include <stdio.h>

#include "trainingFunction.h"
#include "cvFunction.h"
#include "commandLineParser.h"
#include "initCuda.h"

int main(int argc, char **argv)
{
/*	argc = 8;
	argv = new char*[argc];
	argv[1] = "-g";
	argv[2] = "0.382";
	argv[3] = "-c";
	argv[4] = "100";
	argv[5] = "-f";
	argv[6] = "123";
	argv[argc - 1] = "dataset/a9a.txt";
*/
	char fileName[1024];
	char savedFileName[1024];
	Parser parser;
	parser.ParseLine(argc, argv, fileName, savedFileName);

	if(!InitCUDA())
	{
		return 0;
	}

	printf("CUDA initialized.\n");

	if(parser.cross_validation == 1)
	{
		//perform cross validation
		crossValidation(parser.param, fileName, parser.nNumofFeature);
	}
	else
	{
		//perform svm training
		trainSVM(parser.param, fileName, parser.nNumofFeature);
	}

	return 0;
}
