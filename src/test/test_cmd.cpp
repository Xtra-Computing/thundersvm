//
// Created by ss on 18-3-8.
//

#include <thundersvm/cmdparser.h>
#include "gtest/gtest.h"

class CMDTest : public testing::Test {
protected:
    CMDParser cmdParser;
    SvmParam &param = cmdParser.param_cmd;
    static const int max_args = 64;
    const string traing_file_name = "train_dataset";
    const string test_file_name = "test_dataset";
    const string model_file_name = "model";
    int argc;
    char *argv[max_args];

    void read_cmd(string cmd) {
        argc = 0;
        char *p2 = strtok(const_cast<char *>(cmd.c_str()), " ");
        while (p2 && argc < max_args) {
            argv[argc++] = p2;
            p2 = strtok(NULL, " ");
        }
        cmdParser.parse_command_line(argc, argv);
    }
};

TEST_F(CMDTest, test_default) {
    read_cmd("thundersvm-train " + traing_file_name);
    EXPECT_EQ(param.svm_type, 0);
    EXPECT_EQ(param.kernel_type, 2);
    EXPECT_EQ(param.degree, 3);
    EXPECT_NEAR(param.coef0, 0, 1e-5);
    EXPECT_NEAR(param.C, 1, 1e-5);
    EXPECT_NEAR(param.nu, 0.5, 1e-5);
    EXPECT_NEAR(param.p, 0.1, 1e-5);
    EXPECT_NEAR(param.epsilon, 0.001, 1e-5);
    EXPECT_EQ(param.probability, 0);
    EXPECT_EQ(cmdParser.gpu_id, 0);
    EXPECT_EQ(cmdParser.svmtrain_input_file_name, traing_file_name);
}