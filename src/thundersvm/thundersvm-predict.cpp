//
// Created by jiashuai on 17-10-31.
//

#include <thundersvm/util/log.h>
#include <thundersvm/cmdparser.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvr.h>

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
    if (svm_type == "nu_svr") {
        model = new NuSVR();
    }
    model->load_from_file(parser.svmpredict_model_file_name);
    DataSet predict_dataset;
    predict_dataset.load_from_file(parser.svmpredict_input_file);

    vector<real> predict_y;
    predict_y = model->predict(predict_dataset.instances(), 10000);
    real mse = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        mse += (predict_y[i] - predict_dataset.y()[i]) * (predict_y[i] - predict_dataset.y()[i]);
    }
    mse /= predict_y.size();

    LOG(INFO) << "MSE = " << mse;
}
