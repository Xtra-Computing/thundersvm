//
// Created by jiashuai on 17-11-1.
//

#ifndef THUNDERSVM_METRIC_H
#define THUNDERSVM_METRIC_H

#include <thundersvm/thundersvm.h>

class Metric {
public:
    virtual string name() = 0;

    virtual float_type score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) = 0;
};

class Accuracy : public Metric {
    string name() override;

    float_type score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) override;
};

class MSE : public Metric {
    string name() override;

    float_type score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) override;
};

#endif //THUNDERSVM_METRIC_H
