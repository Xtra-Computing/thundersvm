#include <gtest/gtest.h>
#include <thundersvm/dataset.h>
#include <config.h>
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
    SvmModel *model = new NuSVR();
    model->train(dataset, param);

    vector<real> predict_y;
    predict_y = model->predict(dataset.instances(), 100);
    real mse = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        mse += (predict_y[i] - dataset.y()[i]) * (predict_y[i] - dataset.y()[i]);
    }
    mse /= predict_y.size();

    LOG(INFO) << "MSE = " << mse;
}
