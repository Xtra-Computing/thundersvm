//functions for python interface

#include <thundersvm/util/log.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>
#include "thundersvm/cmdparser.h"
#include <thundersvm/svm_interface_api.h>
using std::fstream;
using std::stringstream;
DataSet dataset_python;

extern "C" {
    DataSet* DataSet_new() {return new DataSet();}
    //void DataSet_load_from_python(DataSet *dataset, float *y, char **x, int len) {dataset->load_from_python(y, x, len);}
    void thundersvm_train_sub(DataSet& train_dataset, CMDParser& parser, char* model_file_path){
        SvmModel *model = nullptr;
        switch (parser.param_cmd.svm_type) {
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

        //todo add this to check_parameter method
        if (parser.param_cmd.svm_type == SvmParam::NU_SVC) {
            train_dataset.group_classes();
            for (int i = 0; i < train_dataset.n_classes(); ++i) {
                int n1 = train_dataset.count()[i];
                for (int j = i + 1; j < train_dataset.n_classes(); ++j) {
                    int n2 = train_dataset.count()[j];
                    if (parser.param_cmd.nu * (n1 + n2) / 2 > min(n1, n2)) {
                        printf("specified nu is infeasible\n");
                        return;
                    }
                }
            }
        }
		if (parser.param_cmd.kernel_type != SvmParam::LINEAR)
            if (!parser.gamma_set) {
                parser.param_cmd.gamma = 1.f / train_dataset.n_features();
            }
#ifdef USE_CUDA
        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
#endif

        vector<float_type> predict_y, test_y;
        if (parser.do_cross_validation) {
            predict_y = model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
        } else {
            model->train(train_dataset, parser.param_cmd);
            model->save_to_file(model_file_path);
            LOG(INFO) << "evaluating training score";
            predict_y = model->predict(train_dataset.instances(), -1);
            //predict_y = model->predict(train_dataset.instances(), 10000);
            //test_y = train_dataset.y();
        }
        Metric *metric = nullptr;
        switch (parser.param_cmd.svm_type) {
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
            LOG(INFO) << metric->name() << " = " << metric->score(predict_y, train_dataset.y()) << std::endl;
        }
        delete model;
        delete metric;
        return;
    }
    void thundersvm_train(int argc, char **argv) {
        CMDParser parser;
        parser.parse_command_line(argc, argv);
        /*
        parser.param_cmd.svm_type = SvmParam::NU_SVC;
        parser.param_cmd.kernel_type = SvmParam::RBF;
        parser.param_cmd.C = 100;
        parser.param_cmd.gamma = 0;
        parser.param_cmd.nu = 0.1;
        parser.param_cmd.epsilon = 0.001;
        */
        DataSet train_dataset;
        char input_file_path[1024] = DATASET_DIR;
        char model_file_path[1024] = DATASET_DIR;
//        strcat(input_file_path, "../python/");
//        strcat(model_file_path, "../python/");
        strcpy(input_file_path, parser.svmtrain_input_file_name.c_str());
        strcpy(model_file_path, parser.model_file_name.c_str());
        train_dataset.load_from_file(input_file_path);
        thundersvm_train_sub(train_dataset, parser, model_file_path);
        return;
    }
    void thundersvm_predict_sub(DataSet& predict_dataset, CMDParser& parser, char* model_file_path, char* output_file_path){
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

#ifdef USE_CUDA
        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
#endif

        model->set_max_memory_size_Byte(parser.param_cmd.max_mem_size);
        model->load_from_file(model_file_path);
        file.close();
        file.open(output_file_path, fstream::out);

        vector<float_type> predict_y;
        predict_y = model->predict(predict_dataset.instances(), -1);
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

    void thundersvm_predict(int argc, char **argv){
        CMDParser parser;
        parser.parse_command_line(argc, argv);

        char model_file_path[1024] = DATASET_DIR;
        char predict_file_path[1024] = DATASET_DIR;
        char output_file_path[1024] = DATASET_DIR;
//        strcat(model_file_path, "../python/");
//        strcat(predict_file_path, "../python/");
//        strcat(output_file_path, "../python/");
        strcpy(model_file_path, parser.svmpredict_model_file_name.c_str());
        strcpy(predict_file_path, parser.svmpredict_input_file.c_str());
        strcpy(output_file_path, parser.svmpredict_output_file.c_str());
        DataSet predict_dataset;
        predict_dataset.load_from_file(predict_file_path);
        thundersvm_predict_sub(predict_dataset, parser, model_file_path, output_file_path);
    }

    void load_from_python_interface(float *y, char **x, int len){
        dataset_python.load_from_python(y, x, len);
    }

    void thundersvm_train_after_parse(char **option, int len, char *file_name){
        CMDParser parser;
        parser.parse_python(len, option);
	if(!parser.check_parameter()) return;
        char model_file_path[1024] = DATASET_DIR;
//        strcat(model_file_path, "../python/");
        strcpy(model_file_path, file_name);
        thundersvm_train_sub(dataset_python, parser, model_file_path);
    }
    void thundersvm_predict_after_parse(char *model_file_name, char *output_file_name, char **option, int len){
        CMDParser parser;
        parser.parse_python(len, option);
        char model_file_path[1024] = DATASET_DIR;
        char output_file_path[1024] = DATASET_DIR;
//        strcat(model_file_path, "../python/");
//        strcat(output_file_path, "../python/");
        strcpy(model_file_path, model_file_name);
        strcpy(output_file_path, output_file_name);
        thundersvm_predict_sub(dataset_python, parser, model_file_path, output_file_path);
    }
}
