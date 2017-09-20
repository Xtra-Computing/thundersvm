//
// Created by jiashuai on 17-9-17.
//
#include "thundersvm/dataset.h"

using std::fstream;
using std::stringstream;

DataSet::DataSet() : total_count_(0), n_features_(0) {
}

void DataSet::load_from_file(string file_name) {
    y_.clear();
    index_.clear();
    value_.clear();
    total_count_ = 0;
    n_features_ = 0;
    fstream file;
    file.open(file_name, fstream::in);
    string line;

    while (getline(file, line)) {
        int y, i;
        real v;
        stringstream ss(line);
        ss >> y;
        this->y_.push_back(y);
        this->index_.emplace_back();
        this->value_.emplace_back();
        string tuple;
        size_t n_features = 0;
        while (ss >> tuple) {
            CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2) << "read error, using [index]:[value] format";
            this->index_[total_count_].push_back(i);
            this->value_[total_count_].push_back(v);
            n_features++;
        };
        total_count_++;
        if (n_features > this->n_features_) this->n_features_ = n_features;
    }
    file.close();
    group_classes();
}

const int *DataSet::count() const {
    return count_.data();
}

const int *DataSet::start() const {
    return start_.data();
}

size_t DataSet::n_classes() const {
    return label_.size();
}

const int *DataSet::label() const {
    return label_.data();
}

void DataSet::group_classes() {
    start_.clear();
    count_.clear();
    label_.clear();
    perm_.clear();
    vector<int> dataLabel(y_.size());

    //get the class labels; count the number of instances in each class.
    for (int i = 0; i < y_.size(); ++i) {
        int j;
        for (j = 0; j < label_.size(); ++j) {
            if (y_[i] == label_[j]) {
                count_[j]++;
                break;
            }
        }
        dataLabel[i] = j;
        //if the label is unseen, add it to label vector.
        if (j == label_.size()) {
            label_.push_back(y_[i]);
            count_.push_back(1);
        }
    }

    //logically put instances of the same class consecutively.
    start_.push_back(0);
    for (int i = 1; i < count_.size(); ++i) {
        start_.push_back(start_[i - 1] + count_[i - 1]);
    }
    vector<int> start_copy(start_);
    perm_ = vector<int>(y_.size());
    for (int i = 0; i < y_.size(); ++i) {
        perm_[start_copy[dataLabel[i]]] = i;
        start_copy[dataLabel[i]]++;
    }
}

size_t DataSet::total_count() const {
    return total_count_;
}

size_t DataSet::n_features() const {
    return n_features_;
}

const vector<vector<int>> & DataSet::index() const {
    return this->index_;
}

const vector<vector<real>> & DataSet::value() const {
    return this->value_;
}
