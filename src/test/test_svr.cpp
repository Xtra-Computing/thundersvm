//
// Created by jiashuai on 17-10-5.
//
#include <gtest/gtest.h>
#include "thundersvm/model/svr.h"

TEST(SVRTest, train) {

    DataSet dataSet;
    dataSet.load_from_file("/home/jiashuai/libsvm/abalone_scale");
    SvmParam param;
    param.gamma = 0.25;
    param.C = 10;
    param.p = 0.1;
    SvmModel *model = new SVR(dataSet, param);
    model->train();
}
