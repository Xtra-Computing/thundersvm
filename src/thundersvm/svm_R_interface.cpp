//functions for R interface

#include <thundersvm/util/log.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>
#include "thundersvm/cmdparser.h"
#include <omp.h>
using std::fstream;
using std::stringstream;
extern "C" {
//    void thundersvm_train_R(int *argc, char **argv) {
//        CMDParser parser;
//        parser.parse_command_line(*argc, argv);
//
//        DataSet train_dataset;
//        char input_file_path[1024] = DATASET_DIR;
//        char model_file_path[1024] = DATASET_DIR;
//        strcpy(input_file_path, parser.svmtrain_input_file_name.c_str());
//        strcpy(model_file_path, parser.model_file_name.c_str());
//        train_dataset.load_from_file(input_file_path);
//        SvmModel *model = nullptr;
//        switch (parser.param_cmd.svm_type) {
//            case SvmParam::C_SVC:
//                model = new SVC();
//                break;
//            case SvmParam::NU_SVC:
//                model = new NuSVC();
//                break;
//            case SvmParam::ONE_CLASS:
//                model = new OneClassSVC();
//                break;
//            case SvmParam::EPSILON_SVR:
//                model = new SVR();
//                break;
//            case SvmParam::NU_SVR:
//                model = new NuSVR();
//                break;
//        }
//
//    	//todo add this to check_parameter method
//        if (parser.param_cmd.svm_type == SvmParam::NU_SVC) {
//            train_dataset.group_classes();
//            for (int i = 0; i < train_dataset.n_classes(); ++i) {
//                int n1 = train_dataset.count()[i];
//                for (int j = i + 1; j < train_dataset.n_classes(); ++j) {
//                    int n2 = train_dataset.count()[j];
//                    if (parser.param_cmd.nu * (n1 + n2) / 2 > min(n1, n2)) {
//                        printf("specified nu is infeaclass_weightsible\n");
//                        return;
//                    }
//                }
//            }
//        }
//
//    #ifdef USE_CUDA
//        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
//    #endif
//
//        vector<float_type> predict_y, test_y;
//        if (parser.do_cross_validation) {
//            vector<float_type> test_predict = model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
//            int dataset_size = test_predict.size() / 2;
//    	   test_y.insert(test_y.end(), test_predict.begin(), test_predict.begin() + dataset_size);
//    	   predict_y.insert(predict_y.end(), test_predict.begin() + dataset_size, test_predict.end());
//        } else {
//            model->train(train_dataset, parser.param_cmd);
//            model->save_to_file(model_file_path);
//        	//predict_y = model->predict(train_dataset.instances(), 10000);
//    		//test_y = train_dataset.y();
//        }
//        return;
//    }
//
//    void thundersvm_predict_R(int *argc, char **argv){
//        CMDParser parser;
//        parser.parse_command_line(*argc, argv);
//
//        char model_file_path[1024] = DATASET_DIR;
//        char predict_file_path[1024] = DATASET_DIR;
//        char output_file_path[1024] = DATASET_DIR;
//        strcpy(model_file_path, parser.svmpredict_model_file_name.c_str());
//        strcpy(predict_file_path, parser.svmpredict_input_file.c_str());
//        strcpy(output_file_path, parser.svmpredict_output_file.c_str());
//        fstream file;
//        file.open(model_file_path, std::fstream::in);
//        string feature, svm_type;
//        file >> feature >> svm_type;
//        CHECK_EQ(feature, "svm_type");
//        SvmModel *model = nullptr;
//        Metric *metric = nullptr;
//        if (svm_type == "c_svc") {
//            model = new SVC();
//            metric = new Accuracy();
//        } else if (svm_type == "nu_svc") {
//            model = new NuSVC();
//            metric = new Accuracy();
//        } else if (svm_type == "one_class") {
//            model = new OneClassSVC();
//            //todo determine a metric
//        } else if (svm_type == "epsilon_svr") {
//            model = new SVR();
//            metric = new MSE();
//        } else if (svm_type == "nu_svr") {
//            model = new NuSVR();
//            metric = new MSE();
//        }
//
//    #ifdef USE_CUDA
//        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
//    #endif
//
//        model->load_from_file(model_file_path);
//        file.close();
//        file.open(output_file_path, std::fstream::out);
//        DataSet predict_dataset;
//        predict_dataset.load_from_file(predict_file_path);
//        vector<float_type> predict_y;
//        predict_y = model->predict(predict_dataset.instances(), 10000);
//	    for (int i = 0; i < predict_y.size(); ++i) {
//            file << predict_y[i] << std::endl;
//        }
//        file.close();
//
//        if (metric) {
//            LOG(INFO) << metric->name() << " = " << metric->score(predict_y, predict_dataset.y());
//        }
//    }

    int* train_R(char** dataset, int* kernel, int* svm_type,
                int* degree, char** gamma, double* coef0,
                 double* nu, double* cost, double* epsilon,
                 double* tol, int* probability,
                char** class_weight,  int* weight_length,int* n_fold,
                int* verbose, int* max_iter, int* n_cores, char **model_file){
        int* succeed = new int[1];
        succeed[0] = 1;

        if(*verbose)
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");
        else
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "true");

        if (n_cores[0] > 0) {
            omp_set_num_threads(n_cores[0]);
        } else if (n_cores[0] != -1) {
            LOG(ERROR) << "n_cores must be positive or -1";
        }

        char input_file_path[1024] = DATASET_DIR;
        char model_file_path[1024] = DATASET_DIR;
        strcat(input_file_path, "../R/");
        strcat(model_file_path, "../R/");
        strcpy(input_file_path, dataset[0]);
        if(strcmp(model_file[0], "None") == 0) {
            strcpy(model_file_path, dataset[0]);
            strcat(model_file_path, ".model");
        }
        else
            strcpy(model_file_path, model_file[0]);
        DataSet train_dataset;
        train_dataset.load_from_file(input_file_path);
        SvmModel* model;
        switch (*svm_type){
            case SvmParam::C_SVC:
                model = new SVC();
                break;
            case SvmParam::NU_SVC:
                model = new NuSVC();
                break;
            case SvmParam::ONE_CLASS:
                model = new OneClassSVC();
                break;
            case SvmParam::EPSILON_SVR:
                model = new SVR();
                break;
            case SvmParam::NU_SVR:
                model = new NuSVR();
                break;
        }
        model->set_max_iter(*max_iter);
        //todo add this to check_parameter method
        if (*svm_type == SvmParam::NU_SVC) {
            train_dataset.group_classes();
            for (int i = 0; i < train_dataset.n_classes(); ++i) {
                int n1 = train_dataset.count()[i];
                for (int j = i + 1; j < train_dataset.n_classes(); ++j) {
                    int n2 = train_dataset.count()[j];
                    if (*nu * (n1 + n2) / 2 > min(n1, n2)) {
                        printf("specified nu is infeasible\n");
                        succeed[0] = -1;
                        return succeed;
                    }
                }
            }
        }
        SvmParam param_cmd;
        param_cmd.weight_label = NULL;
        param_cmd.weight = NULL;
        param_cmd.svm_type = static_cast<SvmParam::SVM_TYPE> (*svm_type);
        param_cmd.kernel_type = static_cast<SvmParam::KERNEL_TYPE> (*kernel);
        param_cmd.degree = *degree;
        if((strcmp(gamma[0], "auto") == 0) && (param_cmd.kernel_type != SvmParam::LINEAR)){
            printf("in auto\n");
            param_cmd.gamma = 1.f / train_dataset.n_features();
        }
        else
            param_cmd.gamma = std::atof(gamma[0]);
        param_cmd.coef0 = (float_type)*coef0;
        param_cmd.C = (float_type)*cost;
        param_cmd.nu = (float_type)*nu;
        param_cmd.p = (float_type)*epsilon;
        param_cmd.epsilon = (float_type)*tol;
        param_cmd.probability = *probability;
        if(strcmp(class_weight[0], "None")){
            param_cmd.nr_weight = *weight_length / 2;
            param_cmd.weight = (float_type *) malloc(param_cmd.nr_weight * sizeof(float_type));
            param_cmd.weight_label = (int *) malloc(param_cmd.nr_weight * sizeof(int));
            char* chars_array = strtok(class_weight[0], " ");
            int ind = 0;
            while (chars_array){
                if(ind % 2 == 0){
                    if(strncmp(chars_array, "-w", 2) != 0) {
                        std::cout << "class weight wrong input format!" << std::endl;
                        succeed[0] = -1;
                        return succeed;
                    }
                    param_cmd.weight_label[ind / 2] = atoi(&chars_array[2]);
                }
                else{
                    param_cmd.weight[ind / 2] = atof(chars_array);
                }
                chars_array = strtok(NULL, " ");
                ind++;
            }
        }

        int nr_fold = *n_fold;
        vector<float_type> predict_y;
        bool do_cross_validation = false;
        if (nr_fold != -1) {
            do_cross_validation = true;
            predict_y = model->cross_validation(train_dataset, param_cmd, nr_fold);
        } else {
            model->train(train_dataset, param_cmd);
            LOG(INFO) << "training finished";
            model->save_to_file(model_file_path);
            //   LOG(INFO) << "evaluating training score";
            //   predict_y = model->predict(train_dataset.instances(), -1);
        }
//        vector<float_type> predict_y, test_y;
//        model->train(train_dataset, param_cmd);
//        model->save_to_file(model_file_path);
//        LOG(INFO) << "evaluating training score";
//        predict_y = model->predict(train_dataset.instances(), -1);
        if(do_cross_validation) {
            Metric *metric = nullptr;
            switch (param_cmd.svm_type) {
                case SvmParam::C_SVC:
                case SvmParam::NU_SVC: {
                    metric = new Accuracy();
                    break;
                }
                case SvmParam::EPSILON_SVR:
                case SvmParam::NU_SVR: {
                    metric = new MSE();
                    break;
                }
                case SvmParam::ONE_CLASS: {
                }
            }
            if (metric) {
                std::cout << metric->name() << " = " << metric->score(predict_y, train_dataset.y()) << std::endl;
            }
            delete metric;
        }
        delete model;
        return succeed;
//        model->train(train_dataset, param_cmd);
//        model->save_to_file(model_file_path);
//        LOG(INFO) << "training finished";
//        return succeed;
    }


    void predict_R(char **test_file, char **model_file, char **out_file){
        char model_file_path[1024] = DATASET_DIR;
        char predict_file_path[1024] = DATASET_DIR;
        char output_file_path[1024] = DATASET_DIR;
        strcat(model_file_path, "../R/");
        strcat(predict_file_path, "../R/");
        strcat(output_file_path, "../R/");
        strcpy(model_file_path, model_file[0]);
        strcpy(predict_file_path, test_file[0]);
        strcpy(output_file_path, out_file[0]);
        fstream file;
        file.open(model_file_path, std::fstream::in);
        string feature, svm_type;
        file >> feature >> svm_type;
        CHECK_EQ(feature, "svm_type");
        SvmModel *model = nullptr;
        Metric *metric = nullptr;
        if (svm_type == "c_svc") {
            model = new SVC();
            metric = new Accuracy();
        } else if (svm_type == "nu_svc") {
            model = new NuSVC();
            metric = new Accuracy();
        } else if (svm_type == "one_class") {
            model = new OneClassSVC();
            //todo determine a metric
        } else if (svm_type == "epsilon_svr") {
            model = new SVR();
            metric = new MSE();
        } else if (svm_type == "nu_svr") {
            model = new NuSVR();
            metric = new MSE();
        }

        model->load_from_file(model_file_path);
        file.close();
        file.open(output_file_path, std::fstream::out);
        DataSet predict_dataset;
        predict_dataset.load_from_file(predict_file_path);
        vector<float_type> predict_y;
        predict_y = model->predict(predict_dataset.instances(), 10000);
        for (int i = 0; i < predict_y.size(); ++i) {
            file << predict_y[i] << std::endl;
        }
        file.close();

        if (metric) {
            LOG(INFO) << metric->name() << " = " << metric->score(predict_y, predict_dataset.y());
        }
        delete model;
        delete metric;
    }
}
