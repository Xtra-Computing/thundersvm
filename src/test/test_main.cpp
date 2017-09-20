//
// Created by jiashuai on 17-9-15.
//
#include <thundersvm/syncdata.h>
#include "gtest/gtest.h"
#include "thundersvm/util/log.h"

INITIALIZE_EASYLOGGINGPP
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    return RUN_ALL_TESTS();
}
