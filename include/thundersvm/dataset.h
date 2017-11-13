//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERSVM_DATASET_H
#define THUNDERSVM_DATASET_H

#include "thundersvm.h"
#include "syncdata.h"
class DataSet {
public:
    struct node{
        node(int index, float_type value) : index(index), value(value) {}
        int index;
        float_type value;
    };
    typedef vector<vector<DataSet::node>> node2d;

    DataSet();

    DataSet(const DataSet::node2d &instances, int n_features, const vector<float_type> &y);
    void load_from_file(string file_name);

    void group_classes(bool classification = true);
    size_t n_instances() const;
    size_t n_features() const;
    size_t n_classes() const;

    const vector<int> &count() const;

    const vector<int> &start() const;

    const vector<int> &label() const;

    const vector<float_type> &y() const;
    const node2d & instances() const;
    const node2d instances(int y_i) const;
    const node2d instances(int y_i, int y_j) const;
    const vector<int> original_index() const;
    const vector<int> original_index(int y_i) const;
    const vector<int> original_index(int y_i, int y_j) const;

private:
    vector<float_type> y_;
    node2d instances_;
    size_t total_count_;
    size_t n_features_;
    vector<int> start_; //logical start position of each class
    vector<int> count_; //the number of instances of each class
    vector<int> label_;
    vector<int> perm_;
};
#endif //THUNDERSVM_DATASET_H
