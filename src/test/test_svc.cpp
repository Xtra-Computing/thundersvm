//
// Created by jiashuai on 17-9-21.
//
#include "gtest/gtest.h"
#include "thundersvm/model/SVC.h"
TEST(SVCTest, train){
    DataSet dataSet;
//    dataSet.load_from_file("data/test_dataset.txt");
//    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/iris.scale");
//    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/news20.binary");
//    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/mnist.scale");
    dataSet.load_from_file("/home/jiashuai/mascot_old/dataset/a9a");
    SvmParam param;
    param.gamma = 0.5;
    param.C = 100;
    SvmModel *model = new SVC(dataSet, param);
    model->train();
    vector<int> predict_y;
    predict_y = model->predict(dataSet.instances());
    int n_correct = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        if (predict_y[i] == dataSet.y()[i])
            n_correct++;
    }
    LOG(INFO) << "Accuracy = " << n_correct / (float) dataSet.instances().size() << "(" << n_correct << "/"
              << dataSet.instances().size() << ")";
}