//
// Created by jiashuai on 17-9-17.
//
#include "gtest/gtest.h"
#include "thundersvm/dataset.h"
#include <dataset.h>
TEST(SvmProblemTest, load_dataset){
    DataSet dataSet;
    dataSet.load_from_file(DATASET_DIR "test_dataset.txt");
    dataSet.load_from_file(DATASET_DIR "test_dataset.txt");
    EXPECT_EQ(dataSet.total_count(), 150);
    EXPECT_EQ(dataSet.n_features(), 4);
    EXPECT_EQ(dataSet.count()[0],49);
    EXPECT_EQ(dataSet.count()[1],50);
    EXPECT_EQ(dataSet.count()[2],51);
    EXPECT_EQ(dataSet.start()[0],0);
    EXPECT_EQ(dataSet.start()[1],49);
    EXPECT_EQ(dataSet.start()[2],99);
    EXPECT_EQ(dataSet.label()[0],0);
    EXPECT_EQ(dataSet.label()[1],2);
    EXPECT_EQ(dataSet.label()[2],3);
    EXPECT_EQ(dataSet.n_classes(),3);
    EXPECT_EQ(dataSet.instances(0).size(), 49);
    EXPECT_EQ(dataSet.instances(0,1).size(), 99);
}
