/*
 * commandLineParser.cpp
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "svmParam.h"
#include "kernelType.h"
#include "commandLineParser.h"

int Parser::cross_validation = 1;
int Parser::nr_fold = 0;
int Parser::nNumofFeature = 0;
SVMParam Parser::param;

void print_null(const char *s) {}

/**
 * @brief: parse a line from terminal
 */
void Parser::ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

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

	if(argc != 8)
	{
		HelpInfo();
	}
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			HelpInfo();
		switch(argv[i-1][1])
		{
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
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
			case 'b':
				param.probability = atoi(argv[i]);
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
			case 'f':
				nNumofFeature = atoi(argv[i]);
				if(nNumofFeature < 1)
				{
					HelpInfo();
				}
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				HelpInfo();
		}
	}

//	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		HelpInfo();

	strcpy(pcFileName, argv[i]);

	if(i<argc-1)
		strcpy(pcSavedFileName,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(pcSavedFileName,"%s.model",p);
	}
}


void Parser::HelpInfo()
{
	printf(
	"Usage: mascot -g xx -c xx -f xx training_set_file \n"
	"options:\n"
	"-g gamma : set gamma in kernel function\n"
	"-c cost : set the parameter C \n"
	"-f features : set the number of dimensions of the input data\n"
	);
	exit(1);
}


