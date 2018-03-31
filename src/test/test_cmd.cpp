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
    const float float_max_error = 1e-5;
    int argc;
    char *argv[max_args];

    void read_cmd(string cmd) {
        //reset to default
        argc = 0;
        param = SvmParam();

        //convert command line to argc and argv
        char *p2 = strtok(const_cast<char *>(cmd.c_str()), " ");
        while (p2 && argc < max_args) {
            argv[argc++] = p2;
            p2 = strtok(NULL, " ");
        }
        //parse command
        cmdParser.parse_command_line(argc, argv);
    }
};

TEST_F(CMDTest, test_default) {
    read_cmd("thundersvm-train " + traing_file_name);
    EXPECT_EQ(param.svm_type, SvmParam::C_SVC);
    EXPECT_EQ(param.kernel_type, SvmParam::RBF);
    EXPECT_EQ(param.degree, 3);
    EXPECT_NEAR(param.coef0, 0, float_max_error);
    EXPECT_NEAR(param.C, 1, float_max_error);
    EXPECT_NEAR(param.nu, 0.5, float_max_error);
    EXPECT_NEAR(param.p, 0.1, float_max_error);
    EXPECT_NEAR(param.epsilon, 0.001, float_max_error);
    EXPECT_EQ(param.probability, 0);
    EXPECT_EQ(cmdParser.gpu_id, 0);
    EXPECT_EQ(cmdParser.svmtrain_input_file_name, traing_file_name);
    EXPECT_EQ(cmdParser.model_file_name, traing_file_name + ".model");
    EXPECT_EQ(cmdParser.gamma_set, false);
    EXPECT_EQ(cmdParser.n_cores, -1);
    //todo: check default gamma = 1/n_features
#ifdef USE_CUDA
    //todo: check default device = 0
#endif
}

TEST_F(CMDTest, test_svm_type) {
    read_cmd("thundersvm-train -s 0 " + traing_file_name);
    EXPECT_EQ(param.svm_type, SvmParam::C_SVC);
    read_cmd("thundersvm-train -s 1 " + traing_file_name);
    EXPECT_EQ(param.svm_type, SvmParam::NU_SVC);
    read_cmd("thundersvm-train -s 2 " + traing_file_name);
    EXPECT_EQ(param.svm_type, SvmParam::ONE_CLASS);
    read_cmd("thundersvm-train -s 3 " + traing_file_name);
    EXPECT_EQ(param.svm_type, SvmParam::EPSILON_SVR);
    read_cmd("thundersvm-train -s 4 " + traing_file_name);
    EXPECT_EQ(param.svm_type, SvmParam::NU_SVR);

    EXPECT_DEATH(read_cmd("thundersvm-train -s -1 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -s 5 " + traing_file_name), "");
}

TEST_F(CMDTest, test_kernel_type) {
    read_cmd("thundersvm-train -t 0 " + traing_file_name);
    EXPECT_EQ(param.kernel_type, SvmParam::LINEAR);
    read_cmd("thundersvm-train -t 1 " + traing_file_name);
    EXPECT_EQ(param.kernel_type, SvmParam::POLY);
    read_cmd("thundersvm-train -t 2 " + traing_file_name);
    EXPECT_EQ(param.kernel_type, SvmParam::RBF);
    read_cmd("thundersvm-train -t 3 " + traing_file_name);
    EXPECT_EQ(param.kernel_type, SvmParam::SIGMOID);
    //todo: precomputed kernel
}

TEST_F(CMDTest, test_degree) {
    EXPECT_DEATH(read_cmd("thundersvm-train -d -1 " + traing_file_name), "");
    read_cmd("thundersvm-train -d 0 " + traing_file_name);
    EXPECT_EQ(param.degree, 0);
    read_cmd("thundersvm-train -d 1 " + traing_file_name);
    EXPECT_EQ(param.degree, 1);
    read_cmd("thundersvm-train -d 10 " + traing_file_name);
    EXPECT_EQ(param.degree, 10);
}

TEST_F(CMDTest, test_coef0) {
    read_cmd("thundersvm-train -r -1 " + traing_file_name);
    EXPECT_NEAR(param.coef0, -1, float_max_error);
    read_cmd("thundersvm-train -r -1.5 " + traing_file_name);
    EXPECT_NEAR(param.coef0, -1.5, float_max_error);
    read_cmd("thundersvm-train -r 0 " + traing_file_name);
    EXPECT_NEAR(param.coef0, 0, float_max_error);
    read_cmd("thundersvm-train -r 1 " + traing_file_name);
    EXPECT_NEAR(param.coef0, 1, float_max_error);
    read_cmd("thundersvm-train -r 1.5 " + traing_file_name);
    EXPECT_NEAR(param.coef0, 1.5, float_max_error);
    read_cmd("thundersvm-train -r 10.5 " + traing_file_name);
    EXPECT_NEAR(param.coef0, 10.5, float_max_error);
}

TEST_F(CMDTest, test_cost) {
    EXPECT_DEATH(read_cmd("thundersvm-train -c -1 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -c 0 " + traing_file_name), "");
    read_cmd("thundersvm-train -c 1 " + traing_file_name);
    EXPECT_NEAR(param.C, 1, float_max_error);
    read_cmd("thundersvm-train -c 1.5 " + traing_file_name);
    EXPECT_NEAR(param.C, 1.5, float_max_error);
    read_cmd("thundersvm-train -c 10.5 " + traing_file_name);
    EXPECT_NEAR(param.C, 10.5, float_max_error);
}

TEST_F(CMDTest, test_nu) {
    read_cmd("thundersvm-train -s 1 -n 0.6 " + traing_file_name);
    EXPECT_NEAR(param.nu, 0.6, float_max_error);
    read_cmd("thundersvm-train -s 1 -n 1 " + traing_file_name);
    EXPECT_NEAR(param.nu, 1, float_max_error);
    EXPECT_DEATH(read_cmd("thundersvm-train -s 1 -n -1 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -s 1 -n 0 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -s 1 -n 1.5 " + traing_file_name), "");
}

TEST_F(CMDTest, test_p) {
    read_cmd("thundersvm-train -s 3 -p 0 " + traing_file_name);
    EXPECT_NEAR(param.p, 0, float_max_error);
    read_cmd("thundersvm-train -s 3 -p 1 " + traing_file_name);
    EXPECT_NEAR(param.p, 1, float_max_error);
    read_cmd("thundersvm-train -s 3 -p 1.5 " + traing_file_name);
    EXPECT_NEAR(param.p, 1.5, float_max_error);
    read_cmd("thundersvm-train -s 3 -p 10.5 " + traing_file_name);
    EXPECT_NEAR(param.p, 10.5, float_max_error);
    EXPECT_DEATH(read_cmd("thundersvm-train -s 3 -p -1 " + traing_file_name), "");
}

TEST_F(CMDTest, test_m) {
    read_cmd("thundersvm-train -m 1024 " + traing_file_name);
    EXPECT_EQ(param.max_mem_size, 1024 << 20);
    EXPECT_DEATH(read_cmd("thundersvm-train -m 0 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -m -1 " + traing_file_name), "");
}
TEST_F(CMDTest, test_e) {
    read_cmd("thundersvm-train -e 0.1 " + traing_file_name);
    EXPECT_NEAR(param.epsilon, 0.1, float_max_error);
    read_cmd("thundersvm-train -e 0.0001 " + traing_file_name);
    EXPECT_NEAR(param.epsilon, 0.0001, float_max_error);
    EXPECT_DEATH(read_cmd("thundersvm-train -e -1 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -e 0 " + traing_file_name), "");
}

//todo: check c weight

TEST_F(CMDTest, test_probability_and_cv) {
    read_cmd("thundersvm-train -b 1 " + traing_file_name);
    EXPECT_EQ(param.probability, 1);
    read_cmd("thundersvm-train -v 5 " + traing_file_name);
    EXPECT_EQ(cmdParser.do_cross_validation, 1);
    EXPECT_EQ(cmdParser.nr_fold, 5);
    EXPECT_DEATH(read_cmd("thundersvm-train -v -1 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -v 0 " + traing_file_name), "");
    EXPECT_DEATH(read_cmd("thundersvm-train -v 1 " + traing_file_name), "");
}

#ifdef USE_CUDA
TEST_F(CMDTest, test_gpu_id_){
    read_cmd("thundersvm-train -u 1 " + traing_file_name);
    //todo: check device
}
#endif

TEST_F(CMDTest, test_model_name) {
    read_cmd("thundersvm-train " + traing_file_name + " " + model_file_name);
    EXPECT_EQ(cmdParser.model_file_name, model_file_name);
}

TEST_F(CMDTest, test_predict) {
    read_cmd("thundersvm-predict " + test_file_name + " " + model_file_name + " " + test_file_name + ".predict");
    EXPECT_EQ(cmdParser.svmpredict_input_file, test_file_name);
    EXPECT_EQ(cmdParser.svmpredict_model_file_name, model_file_name);
    EXPECT_EQ(cmdParser.svmpredict_output_file, test_file_name + ".predict");
}

