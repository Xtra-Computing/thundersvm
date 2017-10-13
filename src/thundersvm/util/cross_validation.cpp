//
// Created by jiashuai on 17-10-13.
//
#include <thundersvm/util/cross_validation.h>

real cross_validation(SvmModel &model, const DataSet &dataset, SvmParam param, int n_fold) {
    vector<real> y_test_all;
    vector<real> y_predict_all;
    for (int k = 0; k < n_fold; ++k) {
        DataSet::node2d x_train, x_test;
        vector<real> y_train, y_test;
        for (int i = 0; i < dataset.n_classes(); ++i) {
            int fold_test_count = dataset.count()[i] / n_fold;
            vector<int> class_idx = dataset.original_index(i);
            auto idx_begin = class_idx.begin() + fold_test_count * k;
            auto idx_end = idx_begin;
            while (idx_end != class_idx.end() && idx_end - idx_begin < fold_test_count) idx_end++;
            for (int j: vector<int>(idx_begin, idx_end)) {
                x_test.push_back(dataset.instances()[j]);
                y_test.push_back(dataset.y()[j]);
            }
            class_idx.erase(idx_begin, idx_end);
            for (int j:class_idx) {
                x_train.push_back(dataset.instances()[j]);
                y_train.push_back(dataset.y()[j]);
            }
        }
        DataSet train_dataset(x_train, dataset.n_features(), y_train);
        model.train(train_dataset, param);
        vector<real> y_predict = model.predict(x_test, 1000);
        y_test_all.insert(y_test_all.end(), y_test.begin(), y_test.end());
        y_predict_all.insert(y_predict_all.end(), y_predict.begin(), y_predict.end());
    }
    int n_correct = 0;
    for (int i = 0; i < dataset.total_count(); ++i) {
        if (y_predict_all[i] == y_test_all[i])
            n_correct++;
    }
    return n_correct / (float) dataset.total_count();
}
