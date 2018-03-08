//
// Created by jiashuai on 17-10-13.
//
#include <gtest/gtest.h>
#include <config.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/util/metric.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/model/oneclass_svc.h>
#include <thundersvm/model/nusvr.h>

class CVTest : public testing::Test{
protected:
    DataSet dataSet;
    std::shared_ptr<Metric> metric;
    std::shared_ptr<SvmModel> model;
    vector<float_type> predict_y;

    void SetUp() override {
        dataSet.load_from_file(DATASET_DIR "test_dataset.txt");
    }

    float test_cv(SvmParam param){
        switch (param.svm_type) {
            case SvmParam::C_SVC:
                model.reset(new SVC());
                metric.reset(new Accuracy());
                break;
            case SvmParam::NU_SVC:
                model.reset(new NuSVC());
                metric.reset(new Accuracy());
                break;
            case SvmParam::ONE_CLASS:
                model.reset(new OneClassSVC());
                break;
            case SvmParam::EPSILON_SVR:
                model.reset(new SVR());
                metric.reset(new MSE());
                break;
            case SvmParam::NU_SVR:
                model.reset(new NuSVR());
                metric.reset(new MSE());
                break;
        }
        predict_y = model->cross_validation(dataSet, param, 5);
        return metric->score(predict_y, dataSet.y());
    }
};

TEST_F(CVTest, cv_c_svc){
    SvmParam param;
    param.svm_type = SvmParam::C_SVC;
    param.C = 10;

    param.kernel_type = SvmParam::RBF;
    param.gamma  = 0.125;
    EXPECT_NEAR(test_cv(param), 0.973, 1e-3);
}

TEST_F(CVTest, cv_nu_svc){
    SvmParam param;
    param.svm_type = SvmParam::NU_SVC;

    param.kernel_type = SvmParam::RBF;
    param.gamma  = 0.125;
    EXPECT_NEAR(test_cv(param), 0.660, 1e-3);
}

TEST_F(CVTest, nu_svr){
    SvmParam param;
    param.svm_type = SvmParam::NU_SVR;

    param.gamma  = 0.125;
    EXPECT_NEAR(test_cv(param), 0.045, 1e-3);
}

TEST_F(CVTest, epsilon_svr){
    SvmParam param;
    param.svm_type = SvmParam::EPSILON_SVR;

    param.gamma  = 0.125;
    EXPECT_NEAR(test_cv(param), 0.044, 1e-3);
}

//todo: test one class svc
