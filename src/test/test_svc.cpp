//
// Created by jiashuai on 17-9-21.
//
#include <thundersvm/model/svc.h>
#include "gtest/gtest.h"
#include "dataset.h"

class SVCTest : public ::testing::Test {
protected:
    DataSet train_dataset;
    DataSet test_dataset;
    SvmParam svmParam;
    vector<real> predict_y;

    float load_dataset_and_train(string train_filename, string test_filename, real C, real gamma) {
        train_dataset.load_from_file(train_filename);
        test_dataset.load_from_file(test_filename);
        svmParam.gamma = gamma;
        svmParam.C = C;
        SvmModel *model = new SVC(train_dataset, svmParam);

        model->train();
        predict_y = model->predict(test_dataset.instances(), 10000);
        int n_correct = 0;
        for (int i = 0; i < predict_y.size(); ++i) {
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
//    load_dataset_and_train(DATASET_DIR "test_dataset.txt", DATASET_DIR "test_dataset.txt", 100, 0.5);
//}

TEST_F(SVCTest, a9a) {
    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
                        "a9a", DATASET_DIR
                        "a9a.t", 100, 0.5), 0.826669, 1e-5);
}

//TEST_F(SVCTest, mnist) {
//    load_dataset_and_train(DATASET_DIR "mnist.scale", DATASET_DIR "mnist.scale.t", 10, 0.125);
//}

TEST_F(SVCTest, realsim) {
    EXPECT_NEAR(load_dataset_and_train(DATASET_DIR
                        "real-sim", DATASET_DIR
                        "real-sim", 4, 0.5), 0.997248, 1e-5);
}
