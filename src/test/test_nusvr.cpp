#include <gtest/gtest.h>
#include <thundersvm/dataset.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvr.h>

//
// Created by jiashuai on 17-10-30.
//
TEST(NuSVRTest, train) {

    DataSet dataset;
    dataset.load_from_file(DATASET_DIR "test_dataset.txt");
    SvmParam param;
    param.gamma = 0.25;
    param.C = 10;
    param.epsilon = 0.001;
    param.nu = 0.5;
    param.kernel_type = SvmParam::RBF;
    param.svm_type = SvmParam::NU_SVR;
    SvmModel *model = new NuSVR();
    model->train(dataset, param);
    model->save_to_file(DATASET_DIR "test_dataset.txt.model2");
    SvmModel *new_model = new NuSVR();
    new_model->load_from_file(DATASET_DIR "test_dataset.txt.model2");
    vector<float_type> predict_y;
    predict_y = new_model->predict(dataset.instances(), 100);
    float_type mse = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        mse += (predict_y[i] - dataset.y()[i]) * (predict_y[i] - dataset.y()[i]);
    }
    mse /= predict_y.size();

    LOG(INFO) << "MSE = " << mse;
}
