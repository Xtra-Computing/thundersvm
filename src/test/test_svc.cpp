//
// Created by jiashuai on 17-9-21.
//
#include "gtest/gtest.h"
#include "thundersvm/model/SVC.h"
TEST(SVCTest, train){
    DataSet dataSet;
    dataSet.load_from_file("data/test_dataset.txt");
//    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/a9a");
    SvmParam param;
    param.gamma = 0.5;
    param.C = 10;
    SvmModel *model = new SVC(dataSet, param);
    model->train();
}