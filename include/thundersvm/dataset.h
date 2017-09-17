//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERSVM_DATASET_H
#define THUNDERSVM_DATASET_H

#include "thundersvm.h"
#include "syncdata.h"
class DataSet {
public:
    DataSet();
    void load_from_file(string file_name);
    size_t total_count() const;
    size_t n_features() const;
private:
    void group_classes();
    vector<int> y_;
    vector<vector<int>> index_;
    vector<vector<real>> value_;
    size_t total_count_;
    size_t n_features_;
    vector<int> start_;
    vector<int> count_;
    vector<int> label_;
    vector<int> perm_;
};
#endif //THUNDERSVM_DATASET_H
