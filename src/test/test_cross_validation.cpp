//
// Created by jiashuai on 17-10-13.
//
#include <gtest/gtest.h>
#include <config.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>

TEST(CVTest, cv) {
    DataSet dataset;
    dataset.load_from_file(DATASET_DIR "test_dataset.txt");
//    dataset.load_from_file(DATASET_DIR "mnist.scale");
    SVR model_svr;
    SVC model_svc;
    SvmParam param;
    param.C = 10;
    param.gamma = 0.125;
    param.epsilon = 0.001;
    param.p = 0.01;
    param.nu = 0.5;
    model_svr.cross_validation(dataset, param, 5);
    model_svc.cross_validation(dataset, param, 5);
}
