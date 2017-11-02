//
// Created by jiashuai on 17-10-9.
//
#include <gtest/gtest.h>
#include <thundersvm/model/oneclass_svc.h>
#include <config.h>

TEST(OneClassSVCTest, train) {
    DataSet dataset;
    dataset.load_from_file(DATASET_DIR "test_dataset.txt");
    SvmParam param;
    param.gamma = 0.5;
    param.nu = 0.1;
    param.epsilon = 0.001;
    param.kernel_type = SvmParam::RBF;
    SvmModel *model = new OneClassSVC();
    model->train(dataset, param);

    vector<real> predict_y = model->predict(dataset.instances(), 100);
    int n_pos = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        if (predict_y[i] > 0)
            n_pos++;
    }
//    EXPECT_EQ(n_pos, 75);
}
