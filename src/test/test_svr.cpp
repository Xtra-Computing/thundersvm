//
// Created by jiashuai on 17-10-5.
//
#include <gtest/gtest.h>
#include <thundersvm/config.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>

class SVRTest : public ::testing::Test {
protected:
    SVRTest() : test_dataset() {}

    DataSet train_dataset;
    DataSet test_dataset;
    SvmParam param;
    vector<float_type> predict_y;

    float
    load_dataset_and_train(string train_filename, string test_filename, float_type C, float_type gamma) {
        train_dataset.load_from_file(train_filename);
        test_dataset.load_from_file(test_filename);
        param.gamma = gamma;
        param.C = C;
        param.kernel_type = SvmParam::RBF;
        std::shared_ptr<SvmModel> model;
        model.reset(new SVR());
        model->train(train_dataset, param);
        std::shared_ptr<Metric> metric;
        metric.reset(new MSE());
        predict_y = model->predict(test_dataset.instances(), 100);
        return metric->score(predict_y, test_dataset.y());
    }
};

TEST_F(SVRTest, test_set) {
    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
                        "test_dataset.txt", DATASET_DIR
                        "test_dataset.txt", 100, 0.5), 0.020752, 1e-3);
}
