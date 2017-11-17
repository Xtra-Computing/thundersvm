#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "thundersvm/svmparam.h"
#include "thundersvm/cmdparser.h"

char *save_filename = NULL;
char *restore_filename = NULL;

char svmscale_file_name[1024];

void HelpInfo_svmtrain() {
    printf(
            "Usage (same as LibSVM): thundersvm [options] training_set_file [model_file]\n"
                    "options:\n"
                    "-s svm_type: set type of SVM (default 0)\n"
                    "	0 -- C-SVC		(multi-class classification)\n"
                    "	1 -- nu-SVC		(multi-class classification)\n"
                    "	2 -- one-class SVM\n"
                    "	3 -- epsilon-SVR	(regression)\n"
                    "	4 -- nu-SVR		(regression)\n"
                    "-t kernel_type: set type of kernel function (default 2)\n"
                    "	0 -- linear: u'*v\n"
                    "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
                    "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
                    "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
                    "	4 -- precomputed kernel (kernel values in training_set_file)\n"
                    "-d degree: set degree in kernel function (default 3)\n"
                    "-g gamma: set gamma in kernel function (default 1/num_features)\n"
                    "-r coef0: set coef0 in kernel function (default 0)\n"
                    "-c cost: set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
                    "-n nu: set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
                    "-p epsilon: set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
                    "-e epsilon: set tolerance of termination criterion (default 0.001)\n"
                    "-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
                    "-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
                    "-v n: n-fold cross validation mode\n"
                    "-u n: specify which gpu to use (default 0)\n"
    );
    exit(1);
}

void HelpInfo_svmpredict() {
    printf(
            "Usage: svm-predict [options] test_file model_file output_file\n"
                    "options:\n"
                    "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
    );
    exit(1);
}

void CMDParser::parse_command_line(int argc, char **argv) {
    int i;
    string bin_name = argv[0];
    bin_name = bin_name.substr(bin_name.find_last_of("/") + 1);
    if (bin_name == "thundersvm-train") {
        // parse options
        for (i = 1; i < argc; i++) {
            if (argv[i][0] != '-') break;
            if (++i >= argc)
                HelpInfo_svmtrain();
            switch (argv[i - 1][1]) {

                case 's':
                    param_cmd.svm_type = static_cast<SvmParam::SVM_TYPE>(atoi(argv[i]));
                    break;
                case 't':
                    param_cmd.kernel_type = static_cast<SvmParam::KERNEL_TYPE>(atoi(argv[i]));
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
//                    param_cmd.cache_size = atof(argv[i]);
                    LOG(WARNING) << "setting cache size is not supported";
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
//                    param_cmd.shrinking = atoi(argv[i]);
                    LOG(WARNING) << "shrinking is not supported";
                    break;
                case 'b':
                    param_cmd.probability = atoi(argv[i]);
                    break;
                case 'q':
//                    print_func = &print_null;
                    //todo disable logging
                    i--;
                    break;
                case 'v':
                    do_cross_validation = true;
                    nr_fold = atoi(argv[i]);
                    if (nr_fold < 2) {
                        fprintf(stderr, "n-fold cross validation: n must >= 2\n");
                        HelpInfo_svmtrain();
                    }
                    break;
                case 'w':
                    ++param_cmd.nr_weight;
                    param_cmd.weight_label = (int *) realloc(param_cmd.weight_label, sizeof(int) * param_cmd.nr_weight);
                    param_cmd.weight = (float_type *) realloc(param_cmd.weight, sizeof(double) * param_cmd.nr_weight);
                    param_cmd.weight_label[param_cmd.nr_weight - 1] = atoi(&argv[i - 1][2]);
                    param_cmd.weight[param_cmd.nr_weight - 1] = atof(argv[i]);
                    break;
                case 'u':
                    gpu_id = atoi(argv[i]);
                    break;
                default:
                    fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
                    HelpInfo_svmtrain();
            }
        }

        if (i >= argc)
            HelpInfo_svmtrain();

        strcpy(svmtrain_input_file_name, argv[i]);

        if (i < argc - 1)
            strcpy(model_file_name, argv[i + 1]);
        else {
            char *p = strrchr(argv[i], '/');
            if (p == NULL)
                p = argv[i];
            else
                ++p;
            sprintf(model_file_name, "%s.model", p);
        }
    } else if (bin_name == "thundersvm-predict") {
        FILE *input, *output;
        for (i = 1; i < argc; i++) {
            if (argv[i][0] != '-') break;
            i++;
            switch (argv[i - 1][1]) {
                case 'b':
//                    predict_probability = atoi(argv[i]);
                    break;
                case 'u':
                    gpu_id = atoi(argv[i]);
                    break;
                default:
                    fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
                    HelpInfo_svmpredict();
            }
        }
        if (i >= argc - 2)
            HelpInfo_svmpredict();
        /*
        input = fopen(argv[i], "r");
        if (input == NULL) {
            fprintf(stderr, "can't open input file %s\n", argv[i]);
            exit(1);
        }

        output = fopen(argv[i + 2], "w");
        if (output == NULL) {
            fprintf(stderr, "can't open output file %s\n", argv[i + 2]);
            exit(1);
        }
        */
        strcpy(svmpredict_input_file, argv[i]);
        strcpy(svmpredict_output_file, argv[i + 2]);
        strcpy(svmpredict_model_file_name, argv[i + 1]);
    }
//    else {
//
//        printf("Usage: thundersvm [options] training_set_file [model_file]\n"
//                       "or: thundersvm_predict [options] test_file model_file output_file\n"
//                       "or: thundersvm_scale [options] data_filename\n");
//        exit(0);
//    }
}
