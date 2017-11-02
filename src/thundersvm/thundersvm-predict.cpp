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
INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {
    CMDParser parser;
    parser.parse_command_line(argc, argv);
    fstream file;
    file.open(parser.svmpredict_model_file_name, fstream::in);
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

    CUDA_CHECK(cudaSetDevice(parser.gpu_id));

    model->load_from_file(parser.svmpredict_model_file_name);
    file.close();
    file.open(parser.svmpredict_output_file);
    DataSet predict_dataset;
    predict_dataset.load_from_file(parser.svmpredict_input_file);

    vector<real> predict_y;
    predict_y = model->predict(predict_dataset.instances(), 10000);
    for (int i = 0; i < predict_y.size(); ++i) {
        file << predict_y[i] << std::endl;
    }
    file.close();

    if (metric) {
        LOG(INFO) << metric->name() << " = " << metric->score(predict_y, predict_dataset.y());
    }
}
