//
// Created by jiashuai on 17-9-14.
//


#include <thundersvm/util/log.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>
#include "thundersvm/cmdparser.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {
    CMDParser parser;
    parser.parse_command_line(argc, argv);

    DataSet train_dataset;
    train_dataset.load_from_file(parser.svmtrain_input_file_name);
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

    CUDA_CHECK(cudaSetDevice(parser.gpu_id));

    if (parser.do_cross_validation) {
        model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
        return 0;
    } else {
        model->train(train_dataset, parser.param_cmd);
        model->save_to_file(parser.model_file_name);
    }

    //perform svm testing
    vector<real> predict_y;

    predict_y = model->predict(train_dataset.instances(), 10000);
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
        LOG(INFO) << metric->name() << " = " << metric->score(predict_y, train_dataset.y());
    }
    return 0;
}

