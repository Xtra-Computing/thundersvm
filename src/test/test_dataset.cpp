//
// Created by jiashuai on 17-9-17.
//
#include "gtest/gtest.h"
#include "thundersvm/dataset.h"
TEST(SvmProblemTest, load_dataset){
    DataSet dataSet;
    dataSet.load_from_file("data/test_dataset.txt");
    dataSet.load_from_file("data/test_dataset.txt");
    EXPECT_EQ(dataSet.total_count(), 150);
    EXPECT_EQ(dataSet.n_features(), 4);
}
