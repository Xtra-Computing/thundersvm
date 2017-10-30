#include <src/test/gtest/src/googletest/googletest/include/gtest/gtest.h>
#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/model/nusvc.h>
#include <dataset.h>

//
// Created by jiashuai on 17-10-30.
//
class NuSVCTest : public ::testing::Test {
protected:
    NuSVCTest() : test_dataset() {}

    DataSet train_dataset;
    DataSet test_dataset;
    SvmParam param;
    vector<real> predict_y;

    float load_dataset_and_train(string train_filename, string test_filename, real C, real gamma, real nu) {
        train_dataset.load_from_file(train_filename);
        test_dataset.load_from_file(test_filename);
        param.gamma = gamma;
        param.C = C;
        param.epsilon = 0.001;
        param.nu = nu;
        param.kernel_type = SvmParam::RBF;
//        param.probability = 1;
        SvmModel *model = new NuSVC();

        model->train(train_dataset, param);
        predict_y = model->predict(test_dataset.instances(), 10000);
        int n_correct = 0;
        for (unsigned i = 0; i < predict_y.size(); ++i) {
            if (predict_y[i] == test_dataset.y()[i])
                n_correct++;
        }
        float accuracy = n_correct / (float) test_dataset.instances().size();
        LOG(INFO) << "Accuracy = " << accuracy << "(" << n_correct << "/"
                  << test_dataset.instances().size() << ")\n";
        return accuracy;
    }
};

//TEST_F(SVCTest, test_set) {
//    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
//                        "test_dataset.txt", DATASET_DIR
//                        "test_dataset.txt", 100, 0.5), 0.98, 1e-5);
//}

TEST_F(NuSVCTest, a9a) {
    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
                        "a9a", DATASET_DIR
                        "a9a.t", 100, 0.5, 0.1), 0.826608, 1e-3);
}
//TEST_F(NuSVCTest, a1a1024) {
//    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
//                        "a1a1024", DATASET_DIR
//                        "a1a1024", 100, 0.5), 0.826608, 1e-3);
//}
//
//TEST_F(SVCTest, mnist) {
//    load_dataset_and_train(DATASET_DIR "mnist.scale", DATASET_DIR "mnist.scale.t", 10, 0.125);
//}
//
//TEST_F(SVCTest, realsim) {
//    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
//                        "real-sim", DATASET_DIR
//                        "real-sim", 4, 0.5), 0.997276, 1e-3);
//
