#include <src/test/gtest/src/googletest/googletest/include/gtest/gtest.h>
#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvc.h>
#include <thundersvm/util/metric.h>

//
// Created by jiashuai on 17-10-30.
//
class NuSVCTest : public ::testing::Test {
protected:
    NuSVCTest() : test_dataset() {}

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
        param.svm_type = SvmParam::NU_SVC;
//        param.probability = 1;
        std::shared_ptr<SvmModel> model;
        model.reset(new NuSVC());
        model->train(train_dataset, param);
        predict_y = model->predict(test_dataset.instances(), 10000);
        std::shared_ptr<Metric> metric;
        metric.reset(new Accuracy());
        return metric->score(predict_y, test_dataset.y());
    }
};

TEST_F(NuSVCTest, test_set) {
    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
                        "test_dataset.txt", DATASET_DIR
                        "test_dataset.txt", 100, 0.5, 0.1), 0.98, 1e-3);
}

//TEST_F(NuSVCTest, a9a) {
//    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
//                        "a9a", DATASET_DIR
//                        "a9a.t", 100, 0.5, 0.1), 0.826608, 1e-3);
//}
//
//TEST_F(NuSVCTest, mnist) {
//    load_dataset_and_train(DATASET_DIR "mnist.scale", DATASET_DIR "mnist.scale.t", 10, 0.125, 0.1);
//}

//TEST_F(NuSVCTest, realsim) {
//    load_dataset_and_train(DATASET_DIR "real-sim", DATASET_DIR "real-sim", 4, 0.5, 0.1);
//}

