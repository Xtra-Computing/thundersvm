#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "thundersvm/svmparam.h"
#include <iostream>
using namespace std;
struct SvmParam param_cmd;
char *line = NULL;
int max_line_len = 1024;
double lower=-1.0,upper=1.0,y_lower,y_upper;
int y_scaling = 0;

char *save_filename = NULL;
char *restore_filename = NULL;

char svmtrain_input_file_name[1024];
char svmtrain_model_file_name[1024];
char svmpredict_input_file[1024];
char svmpredict_output_file[1024];
char svmpredict_model_file_name[1024];
char svmscale_file_name[1024];

int cross_validation;
int nr_fold;
int predict_probability=0;
void print_null(const char *s) {};
static int (*info)(const char *fmt,...) = &printf;
void HelpInfo_svmtrain()
{
	/*
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	);
	exit(1);
	*/
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	);
	exit(1);
}
void HelpInfo_svmpredict()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	);
	exit(1);
}
void HelpInfo_svmscale()
{
	printf(
	"Usage: svm-scale [options] data_filename\n"
	"options:\n"
	"-l lower : x scaling lower limit (default -1)\n"
	"-u upper : x scaling upper limit (default +1)\n"
	"-y y_lower y_upper : y scaling limits (default: no y scaling)\n"
	"-s save_filename : save scaling parameters to save_filename\n"
	"-r restore_filename : restore scaling parameters from restore_filename\n"
	);
	exit(1);
}

void InitParam()
{
	param_cmd.svm_type = C_SVC;
	param_cmd.kernel_type = RBF;
	param_cmd.degree = 3;
	param_cmd.gamma = 0;	// 1/num_features
	param_cmd.coef0 = 0;
	param_cmd.nu = 0.5;
	param_cmd.cache_size = 100;
	param_cmd.C = 1;
	param_cmd.epsilon = 1e-3;
	param_cmd.p = 0.1;
	param_cmd.shrinking = 1;
	param_cmd.probability = 0;
	param_cmd.nr_weight = 0;
	param_cmd.weight_label = NULL;
	param_cmd.weight = NULL;
}

void parse_command_line(int argc, char **argv)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	InitParam();
	if(strcmp(argv[1], "svmtrain") == 0){
		// parse options
		param_cmd.task_type = svmTrain;
		for(i = 2; i < argc; i++)
		{
			if(argv[i][0] != '-') break;
			if(++i >= argc)
				HelpInfo_svmtrain();
			switch(argv[i-1][1])
			{

			case 's':
				param_cmd.svm_type = atoi(argv[i]);
				break;
			case 't':
				param_cmd.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param_cmd.degree = atoi(argv[i]);
				break;
			case 'g':
				param_cmd.gamma = atof(argv[i]);
				break;
			case 'r':
				param_cmd.coef0 = atof(argv[i]);
				break;
			case 'n':
				param_cmd.nu = atof(argv[i]);
				break;
			case 'm':
				param_cmd.cache_size = atof(argv[i]);
				break;
			case 'c':
				param_cmd.C = atof(argv[i]);
				break;
			case 'e':
				param_cmd.epsilon = atof(argv[i]);
				break;
			case 'p':
				param_cmd.p = atof(argv[i]);
				break;
			case 'h':
				param_cmd.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param_cmd.probability = atoi(argv[i]);
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
					HelpInfo_svmtrain();
				}
				break;
			case 'w':
				++param_cmd.nr_weight;
				param_cmd.weight_label = (int *)realloc(param_cmd.weight_label,sizeof(int)*param_cmd.nr_weight);
				param_cmd.weight = (double *)realloc(param_cmd.weight,sizeof(double)*param_cmd.nr_weight);
				param_cmd.weight_label[param_cmd.nr_weight-1] = atoi(&argv[i-1][2]);
				param_cmd.weight[param_cmd.nr_weight-1] = atof(argv[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				HelpInfo_svmtrain();
			}
		}

		if(i>=argc)
			HelpInfo_svmtrain();

		strcpy(svmtrain_input_file_name, argv[i]);

		if(i<argc-1)
			strcpy(svmtrain_model_file_name,argv[i+1]);
		else
		{
			char *p = strrchr(argv[i],'/');
			if(p==NULL)
				p = argv[i];
			else
				++p;
			sprintf(svmtrain_model_file_name,"%s.model",p);
		}
	}
	else if(strcmp(argv[1], "svmpredict") == 0)
	{
		FILE *input, *output;
		param_cmd.task_type = svmPredict;
		for(i = 2; i < argc; i++)
		{
			if(argv[i][0] != '-') break;
			i++;
			switch(argv[i-1][1])
			{
				case 'b':
					predict_probability = atoi(argv[i]);
					break;
				default:
					fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
					HelpInfo_svmpredict();
			}
		}
		if(i >= argc - 2)
			HelpInfo_svmpredict();
		
		input = fopen(argv[i],"r");
		if(input == NULL)
		{
			fprintf(stderr,"can't open input file %s\n",argv[i]);
			exit(1);
		}

		output = fopen(argv[i+2],"w");
		if(output == NULL)
		{
			fprintf(stderr,"can't open output file %s\n",argv[i+2]);
			exit(1);
		}
		strcpy(svmpredict_input_file, argv[i]);
		strcpy(svmpredict_output_file, argv[i+2]);
		strcpy(svmpredict_model_file_name, argv[i+1]);
	}
	else if(strcmp(argv[1], "svmscale") == 0)
	{
		FILE *fp, *fp_restore = NULL;
		char *save_filename = NULL;
		char *restore_filename = NULL;
		for(i = 2; i < argc; i++)
		{
			if(argv[i][0] != '-') break;
			i++;
			switch(argv[i-1][1])
			{
				case 'l': lower = atof(argv[i]); break;
				case 'u': upper = atof(argv[i]); break;
				case 'y':
					y_lower = atof(argv[i]);
					++i;
					y_upper = atof(argv[i]);
					y_scaling = 1;
					break;
				case 's': save_filename = argv[i]; break;
				case 'r': restore_filename = argv[i]; break;
				default:
					fprintf(stderr,"unknown option\n");
					HelpInfo_svmscale();
			}
		}
		if(!(upper > lower) || (y_scaling && !(y_upper > y_lower)))
		{
			fprintf(stderr,"inconsistent lower/upper specification\n");
			exit(1);
		}
		if(restore_filename && save_filename)
		{
			fprintf(stderr,"cannot use -r and -s simultaneously\n");
			exit(1);
		}
		if(argc != i+1) 
			HelpInfo_svmscale();
		strcpy(svmscale_file_name, argv[i]);
	}
}