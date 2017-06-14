/*
 * commandLineParser.cpp
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "../svm-shared/svmParam.h"
#include "kernelType.h"
#include "SVMCmdLineParser.h"

int SVMCmdLineParser::task_type = 1;
bool SVMCmdLineParser::compute_training_error = false;
int SVMCmdLineParser::nr_fold = 0;
string SVMCmdLineParser::testSetName = "";
SVMParam SVMCmdLineParser::param;

void SVMCmdLineParser::InitParam(){
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}

/**
 * @brief: parse a line from terminal
 */
bool SVMCmdLineParser::HandleOption(char c, char *pcOptionValue)
{
	switch(c)
	{
	case 'g':
		param.gamma = atof(pcOptionValue);
		return true;
	case 'c':
		param.C = atof(pcOptionValue);
		return true;
	case 'b':
		param.probability = atoi(pcOptionValue);
		return true;
	case 'o':
		task_type = atoi(pcOptionValue);
		return true;
	case 'r':
		compute_training_error = atoi(pcOptionValue);	//boolean variable
		return true;
	case 'e':
		testSetName = pcOptionValue;
		return true;
	default:
		return false;
		/*
		 case 's':
		 param.svm_type = atoi(argv[i]);
		 break;
		 case 't':
		 param.kernel_type = atoi(argv[i]);
		 break;
		 case 'd':
		 param.degree = atoi(argv[i]);
		 break;
		 case 'r':
		 param.coef0 = atof(argv[i]);
		 break;
		 case 'n':
		 param.nu = atof(argv[i]);
		 break;
		 case 'm':
		 param.cache_size = atof(argv[i]);
		 break;
		 case 'e':
		 param.eps = atof(argv[i]);
		 break;
		 case 'p':
		 param.p = atof(argv[i]);
		 break;
		 case 'h':
		 param.shrinking = atoi(argv[i]);
		 break;
		 case 'q':
		 print_func = &print_null;
		 i--;
		 break;
		 case 'v':
		 cross_validation = 1;
		 nr_fold = atoi(argv[i]);
		 if(nr_fold < 2)
		 {
		 fprintf(stderr,"n-fold cross validation: n must >= 2\n");
		 HelpInfo();
		 }
		 break;
		 case 'w':
		 ++param.nr_weight;
		 param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
		 param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
		 param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
		 param.weight[param.nr_weight-1] = atof(argv[i]);
		 break;
		 */
	}
}


void SVMCmdLineParser::HelpInfo()
{
	printf(
	"Usage: mascot -b xx -c xx -g xx -o xx -r xx -f xx training_set_file \n"
	"options:\n"
	"-b: enable probability output\n"
	"-c cost : set the parameter C \n"
	"-g gamma : set gamma in kernel function\n"
	"-o task type: choose cross-validation, training, or param evaluation\n"
	"-r: evaluate training error\n"
	);
	exit(1);
}


