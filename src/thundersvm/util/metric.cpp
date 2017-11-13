//
// Created by jiashuai on 17-11-1.
//
#include <thundersvm/util/metric.h>

string Accuracy::name() {
    return "Accuracy";
}

float_type Accuracy::score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) {
    int n_correct = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        if (predict_y[i] == ground_truth_y[i])
            n_correct++;
    }
    float accuracy = n_correct / (float) ground_truth_y.size();
    return accuracy;
}

string MSE::name() {
    return "Mean Squared Error";
}

float_type MSE::score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) {
    float_type mse = 0;
    for (int i = 0; i < predict_y.size(); ++i) {
        mse += (predict_y[i] - ground_truth_y[i]) * (predict_y[i] - ground_truth_y[i]);
    }
    mse /= predict_y.size();
    return mse;
}
