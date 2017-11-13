//
// Created by jiashuai on 17-10-5.
//
#include <gtest/gtest.h>
#include <config.h>
#include <thundersvm/model/nusvr.h>

TEST(SVRTest, train) {

    DataSet dataset;
    dataset.load_from_file(DATASET_DIR "test_dataset.txt");
//    dataset.load_from_file(DATASET_DIR "E2006.train");
    SvmParam param;
    param.gamma = 0.25;
    param.C = 10;
    param.p = 0.1;
    param.epsilon = 0.001;
    param.nu = 0.5;
    param.kernel_type = SvmParam::RBF;
    param.svm_type = SvmParam::EPSILON_SVR;
    SvmModel *model = new SVR();
    model->train(dataset, param);
    model->save_to_file(DATASET_DIR "test_dataset.txt.model");
    SvmModel *new_model = new SVR();
    new_model->load_from_file(DATASET_DIR "test_dataset.txt.model");
    vector<float_type> predict_y;
    predict_y = new_model->predict(dataset.instances(), 100);
    float_type mse = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        mse += (predict_y[i] - dataset.y()[i]) * (predict_y[i] - dataset.y()[i]);
    }
    mse /= predict_y.size();

    LOG(INFO) << "MSE = " << mse;
    EXPECT_NEAR(mse, 0.03097, 1e-4);
}
