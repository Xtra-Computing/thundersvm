//
// Created by jiashuai on 17-11-1.
//

#ifndef THUNDERSVM_METRIC_H
#define THUNDERSVM_METRIC_H

#include <thundersvm/thundersvm.h>

/**
 * @brief metric for evaluation model
 */
class Metric {
public:
    virtual string name() = 0;

    virtual float_type score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) = 0;

    virtual ~Metric() = default;
};

/**
 * @brief Accuracy
 */
class Accuracy : public Metric {
public:
    ~Accuracy() override = default;
    string name() override;

    /**
     * \f$\frac{#\text{correct}}{#\text{all}}\f$
     * @param predict_y
     * @param ground_truth_y
     * @return accuracy score
     */
    float_type score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) override;
};

/**
 * @brief Mean Squared Error
 */
class MSE : public Metric {
public:
    ~MSE() override = default;
    string name() override;

    /**
     * \f$\text{MSE} = \frac{1}{n}\sum_{i}{(y_i-y^*_i)}^2\f$, where \f$y\f$ is predicted and \f$y^*\f$ is the ground
     * truth, \f$n\f$ is the number of instances.
     * @param predict_y
     * @param ground_truth_y
     * @return MSE
     */
    float_type score(const vector<float_type> &predict_y, const vector<float_type> &ground_truth_y) override;
};

#endif //THUNDERSVM_METRIC_H
