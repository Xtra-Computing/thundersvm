//
// Created by jiashuai on 17-10-31.
//

#include <thundersvm/util/log.h>
#include <thundersvm/cmdparser.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/util/metric.h>

using std::fstream;

int main(int argc, char **argv) {
    try {
        CMDParser parser;
        parser.parse_command_line(argc, argv);
        fstream file;
        file.open(parser.svmpredict_model_file_name, fstream::in);
        std::cout << parser.svmpredict_model_file_name << "\n";
        string feature, svm_type;
        file >> feature >> svm_type;
        std::cout << feature << "; " << svm_type << "\n";
        CHECK_EQ(feature, "svm_type");
        std::shared_ptr<SvmModel> model;
        std::shared_ptr<Metric> metric;
        if (svm_type == "c_svc") {
            model.reset(new SVC());
            metric.reset(new Accuracy());
        } else if (svm_type == "nu_svc") {
            model.reset(new NuSVC());
            metric.reset(new Accuracy());
        } else if (svm_type == "one_class") {
            model.reset(new OneClassSVC());
            //todo determine a metric
        } else if (svm_type == "epsilon_svr") {
            model.reset(new SVR());
            metric.reset(new MSE());
        } else if (svm_type == "nu_svr") {
            model.reset(new NuSVR());
            metric.reset(new MSE());
        }

#ifdef USE_CUDA
        CUDA_CHECK(cudaSetDevice(parser.gpu_id));
#endif

        model->load_from_file(parser.svmpredict_model_file_name);
        file.close();
        file.open(parser.svmpredict_output_file);
        DataSet predict_dataset;
        predict_dataset.load_from_file(parser.svmpredict_input_file);

        vector<float_type> predict_y;
        predict_y = model->predict(predict_dataset.instances(), 10000);
        for (int i = 0; i < predict_y.size(); ++i) {
            file << predict_y[i] << std::endl;
        }
        file.close();

        if (metric) {
            LOG(INFO) << metric->name() << " = " << metric->score(predict_y, predict_dataset.y());
        }
    }
    catch (std::bad_alloc &) {
        LOG(FATAL) << "out of host memory";
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
