//
// Created by jiashuai on 17-9-15.
//
#include "thundersvm/thundersvm.h"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
//    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");
    return RUN_ALL_TESTS();
}
