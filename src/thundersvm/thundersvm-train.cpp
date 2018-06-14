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

#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif

int main(int argc, char **argv) {
    try {
		el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
        CMDParser parser;
        parser.parse_command_line(argc, argv);
        DataSet train_dataset;
        train_dataset.load_from_file(parser.svmtrain_input_file_name);
        std::shared_ptr<SvmModel> model;
        switch (parser.param_cmd.svm_type) {
            case SvmParam::C_SVC:
                model.reset(new SVC());
                LOG(INFO) << "training C-SVC";
                LOG(INFO) << "C = " << parser.param_cmd.C;
                break;
            case SvmParam::NU_SVC:
                model.reset(new NuSVC());
                LOG(INFO) << "training nu-SVC";
                LOG(INFO) << "nu = " << parser.param_cmd.nu;
                break;
            case SvmParam::ONE_CLASS:
                model.reset(new OneClassSVC());
                LOG(INFO) << "training one-class SVM";
                LOG(INFO) << "C = " << parser.param_cmd.C;
                break;
            case SvmParam::EPSILON_SVR:
                model.reset(new SVR());
                LOG(INFO) << "training epsilon-SVR";
                LOG(INFO) << "C = " << parser.param_cmd.C << " p = " << parser.param_cmd.p;
                break;
            case SvmParam::NU_SVR:
                model.reset(new NuSVR());
                LOG(INFO) << "training nu-SVR";
                LOG(INFO) << "nu = " << parser.param_cmd.nu;
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
                        return 1;
                    }
                }
            }
        }
        if (parser.param_cmd.kernel_type != SvmParam::LINEAR)
            if (!parser.gamma_set) {
                parser.param_cmd.gamma = 1.f / train_dataset.n_features();
                LOG(WARNING) << "using default gamma=" << parser.param_cmd.gamma;
            } else {
                LOG(INFO) << "gamma = " << parser.param_cmd.gamma;
            }

#ifdef USE_CUDA
        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
#endif

        vector<float_type> predict_y;
        if (parser.do_cross_validation) {
            predict_y = model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
        } else {
            model->train(train_dataset, parser.param_cmd);
            LOG(INFO) << "training finished";
            model->save_to_file(parser.model_file_name);
         //   LOG(INFO) << "evaluating training score";
         //   predict_y = model->predict(train_dataset.instances(), -1);
        }

        //perform svm testing
        if(parser.do_cross_validation) {
            std::shared_ptr<Metric> metric;
            switch (parser.param_cmd.svm_type) {
                case SvmParam::C_SVC:
                case SvmParam::NU_SVC: {
                    metric.reset(new Accuracy());
                    break;
                }
                case SvmParam::EPSILON_SVR:
                case SvmParam::NU_SVR: {
                    metric.reset(new MSE());
                    break;
                }
                case SvmParam::ONE_CLASS: {
                }
            }
            if (metric) {
                std::cout << "Cross " << metric->name() << " = " << metric->score(predict_y, train_dataset.y()) << std::endl;
            }
        }

    }
    catch (std::bad_alloc &) {
        LOG(FATAL) << "out of memory, you may try \"-m memory size\" to constrain memory usage";
        exit(EXIT_FAILURE);
    }
    catch (std::exception const &x) {
        LOG(FATAL) << x.what();
        exit(EXIT_FAILURE);
    }
    catch (...) {
        LOG(FATAL) << "unknown error";
        exit(EXIT_FAILURE);
    }
}

