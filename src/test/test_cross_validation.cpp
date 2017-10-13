//
// Created by jiashuai on 17-10-13.
//
#include <gtest/gtest.h>
#include <thundersvm/util/cross_validation.h>
#include <dataset.h>
#include <thundersvm/model/svc.h>

TEST(CVTest, cv) {
    DataSet dataset;
    dataset.load_from_file(DATASET_DIR "test_dataset.txt");
//    dataset.load_from_file(DATASET_DIR "mnist.scale");
    dataset.group_classes();
    SVC model;
    SvmParam param;
    param.C = 10;
    param.gamma = 0.125;
    param.epsilon = 0.001;
    real score = cross_validation(model, dataset, param, 5);
    LOG(INFO) << "CV Score = " << score;
}
