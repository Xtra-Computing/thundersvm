#include <gtest/gtest.h>
#include <thundersvm/dataset.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvr.h>
#include <thundersvm/util/metric.h>

//
// Created by jiashuai on 17-10-30.
//

class NuSVRTest : public ::testing::Test {
protected:
    NuSVRTest() : test_dataset() {}

    DataSet train_dataset;
    DataSet test_dataset;
    SvmParam param;
    vector<float_type> predict_y;

    float
    load_dataset_and_train(string train_filename, string test_filename, float_type C, float_type gamma, float_type nu) {
        train_dataset.load_from_file(train_filename);
        test_dataset.load_from_file(test_filename);
        param.gamma = gamma;
        param.C = C;
        param.nu = nu;
        param.kernel_type = SvmParam::RBF;
        std::shared_ptr<SvmModel> model;
        model.reset(new NuSVR());
        model->train(train_dataset, param);
        std::shared_ptr<Metric> metric;
        metric.reset(new MSE());
        predict_y = model->predict(test_dataset.instances(), 100);
        return metric->score(predict_y, test_dataset.y());
    }
};

TEST_F(NuSVRTest, test_set) {
    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
                        "test_dataset.txt", DATASET_DIR
                        "test_dataset.txt", 100, 0.5, 0.1), 0.028369, 1.2e-5);
}
