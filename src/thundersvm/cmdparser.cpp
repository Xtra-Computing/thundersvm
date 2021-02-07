/*
    Since the format of parameters are same as libsvm, we reference and modify command line parser source code of libsvm in this file.
    
    Copyright (c) 2000-2014 Chih-Chung Chang and Chih-Jen Lin
    All rights reserved.
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "thundersvm/svmparam.h"
#include "thundersvm/cmdparser.h"
#include <omp.h>

void HelpInfo_svmtrain() {
    LOG(INFO) <<
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
//                    "	4 -- precomputed kernel (kernel values in training_set_file)\n"
            "-d degree: set degree in kernel function (default 3)\n"
            "-g gamma: set gamma in kernel function (default 1/num_features)\n"
            "-r coef0: set coef0 in kernel function (default 0)\n"
            "-c cost: set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
            "-n nu: set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
            "-p epsilon: set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
            "-m memory size: constrain the maximum memory size (MB) that thundersvm uses (default 8192)\n"
            "-e epsilon: set tolerance of termination criterion (default 0.001)\n"
            "-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
            "-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
            "-v n: n-fold cross validation mode\n"
            "-u n: specify which gpu to use (default 0)\n"
            "-o n: set the number of cpu cores to use, -1 for maximum(default -1)\n"
            "-q: quiet mode";
    exit(1);
}

void HelpInfo_svmpredict() {
    LOG(INFO) <<
              "Usage: svm-predict [options] test_file model_file output_file\n"
                      "options:\n"
                      //todo probability prediction
//                      "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
                      "-m memory size: constrain the maximum memory size (MB) that thundersvm uses\n"
                      "-u n: specify which gpu to use (default 0)\n";
    exit(1);
}

void CMDParser::parse_command_line(int argc, char **argv) {
    param_cmd.weight_label = NULL;
    param_cmd.weight = NULL;
    int i;
    string bin_name = argv[0];
#ifdef _WIN32
    bin_name = bin_name.substr(bin_name.find_last_of("\\") + 1);
#else
    bin_name = bin_name.substr(bin_name.find_last_of("/") + 1);
#endif
    bool quiet = false;
    if (bin_name == "thundersvm-train" || bin_name == "thundersvm-train.exe") {
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
                case 'g': {//handle fraction
                    string str_argv(argv[i]);
                    int slash = (int) str_argv.find("/", 0);
                    if (slash != string::npos) {
                        float_type numerator = atof(str_argv.substr(0, slash).c_str());
                        float_type denominator = atof(str_argv.substr(slash + 1).c_str());
                        param_cmd.gamma = numerator / denominator;
                    } else
                        param_cmd.gamma = atof(argv[i]);
                }
                    gamma_set = true;
                    break;
                case 'r':
                    param_cmd.coef0 = atof(argv[i]);
                    break;
                case 'n':
                    param_cmd.nu = atof(argv[i]);
                    break;
                case 'm':
                    param_cmd.max_mem_size = static_cast<size_t>(max(atoi(argv[i]), 0)) << 20;//MB to Byte
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
                case 'o':
                    n_cores = atoi(argv[i]);
                    break;
                case 'q':
                    quiet = true;
                    i--;
                    break;
                case 'v':
                    do_cross_validation = true;
                    nr_fold = atoi(argv[i]);
                    if (nr_fold < 2) {
                        LOG(ERROR) << "n-fold cross validation: n must >= 2";
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
                    LOG(ERROR) << "Unknown option: " + string(argv[i - 1]);
                    HelpInfo_svmtrain();
            }
        }
        if (n_cores > 0) {
            omp_set_num_threads(n_cores);
        } else if (n_cores != -1) {
            LOG(ERROR) << "the number of cpu cores must be positive or -1";
        }
        if (i >= argc)
            HelpInfo_svmtrain();

        if (!check_parameter()) HelpInfo_svmtrain();

        svmtrain_input_file_name = argv[i];

        if (i < argc - 1)
            model_file_name = argv[i + 1];
        else {
            char *p = strrchr(argv[i], '/');
            if (p == NULL)
                p = argv[i];
            else
                ++p;
            model_file_name = p;
            model_file_name += ".model";
        }
    } else if (bin_name == "thundersvm-predict" || bin_name == "thundersvm-predict.exe") {
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
                case 'o':
                    n_cores = atoi(argv[i]);
                    break;
                case 'm':
                    param_cmd.max_mem_size = static_cast<size_t>(max(atoi(argv[i]), 0)) << 20;//MB to Byte
                    break;
                default:
                    fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
                    HelpInfo_svmpredict();
            }
        }
        if (n_cores > 0) {
            omp_set_num_threads(n_cores);
        } else if (n_cores != -1) {
            LOG(ERROR) << "the number of cpu cores must be positive or -1";
        }
        if (i >= argc - 2)
            HelpInfo_svmpredict();
        svmpredict_input_file = argv[i];
        svmpredict_output_file = argv[i + 2];
        svmpredict_model_file_name = argv[i + 1];
    }
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, quiet ? "false" : "true");
}

void CMDParser::parse_python(int argc, char **argv) {
    //todo: refactor this function, since it overlaps parse_command_line too much
    param_cmd.weight_label = NULL;
    param_cmd.weight = NULL;
    bool quiet = false;
    int i;
    for (i = 0; i < argc; i++) {
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
				gamma_set = true;
                break;
            case 'r':
                param_cmd.coef0 = atof(argv[i]);
                break;
            case 'n':
                param_cmd.nu = atof(argv[i]);
                break;
            case 'm':
                param_cmd.max_mem_size = static_cast<size_t>(max(atoi(argv[i]), 0)) << 20;//MB to Byte
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
            case 'o':
                n_cores = atoi(argv[i]);
                break;
            case 'q':
                quiet = true;
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
    if (n_cores > 0) {
        omp_set_num_threads(n_cores);
    } else if (n_cores != -1) {
        LOG(ERROR) << "the number of cpu cores must be positive or -1";
    }
    if (i > argc)
        HelpInfo_svmtrain();
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, quiet ? "false" : "true");
}

bool CMDParser::check_parameter() {
    SvmParam::SVM_TYPE svm_type = param_cmd.svm_type;
    if (svm_type < SvmParam::C_SVC || svm_type > SvmParam::NU_SVR) {
        LOG(ERROR) << "unknown svm type";
        return false;
    }
    SvmParam::KERNEL_TYPE kernel_type = param_cmd.kernel_type;
    if (kernel_type < SvmParam::LINEAR || kernel_type > SvmParam::PRECOMPUTED) {
        LOG(ERROR) << "unknown kernel type";
        return false;
    }
    if (param_cmd.gamma < 0) {
        LOG(ERROR) << "gamma < 0";
        return false;
    }
    if (param_cmd.degree < 0) {
        LOG(ERROR) << "degree of polynomial kernel < 0";
        return false;
    }
    if (param_cmd.epsilon <= 0) {
        LOG(ERROR) << "epsilon <= 0";
        return false;
    }
    if (svm_type == SvmParam::C_SVC || svm_type == SvmParam::EPSILON_SVR || svm_type == SvmParam::NU_SVR)
        if (param_cmd.C <= 0) {
            LOG(ERROR) << "C <= 0";
            return false;
        }
    if (svm_type == SvmParam::NU_SVC || svm_type == SvmParam::ONE_CLASS || svm_type == SvmParam::NU_SVR)
        if (param_cmd.nu <= 0 || param_cmd.nu > 1) {
            LOG(ERROR) << "nu <= 0 or nu > 1";
            return false;
        }
    if (svm_type == SvmParam::EPSILON_SVR)
        if (param_cmd.p < 0) {
            LOG(ERROR) << "p < 0";
            return false;
        }
    if (param_cmd.max_mem_size <= 0) {
        LOG(ERROR) << "max memory size <= 0";
        return false;
    }
    if (param_cmd.probability != 0 && param_cmd.probability != 1) {
        LOG(ERROR) << "probability != 0 and probability != 1";
        return false;
    }
    if (param_cmd.probability == 1 &&
        (svm_type == SvmParam::ONE_CLASS || svm_type == SvmParam::EPSILON_SVR || svm_type == SvmParam::NU_SVR)) {
        LOG(ERROR) << "one-class SVM and SVR probability output not supported yet";
        return false;
    }
    return true;
}
