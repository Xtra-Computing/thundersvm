//
// Created by jiashuai on 17-9-17.
//
#include "thundersvm/dataset.h"

using std::fstream;
using std::stringstream;

DataSet::DataSet() : total_count_(0), n_features_(0) {}

DataSet::DataSet(const DataSet::node2d &instances, int n_features, const vector<real> &y) :
        instances_(instances), n_features_(n_features), y_(y), total_count_(instances_.size()) {}
void DataSet::load_from_file(string file_name) {
    y_.clear();
    instances_.clear();
    total_count_ = 0;
    n_features_ = 0;
    fstream file;
    file.open(file_name, fstream::in);
    CHECK(file.is_open()) << "file " << file_name << " not found";
    string line;

    while (getline(file, line)) {
        real y;
        int i;
        real v;
        stringstream ss(line);
        ss >> y;
        this->y_.push_back(y);
        this->instances_.emplace_back();//reserve space for an instance
        string tuple;
        while (ss >> tuple) {
            CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2) << "read error, using [index]:[value] format";
            this->instances_[total_count_].emplace_back(i, v);
            if (i > this->n_features_) this->n_features_ = i;
        };
        total_count_++;
    }
    file.close();
}

const vector<int> &DataSet::count() const {//return the number of instances of each class
    return count_;
}

const vector<int> &DataSet::start() const {
    return start_;
}

size_t DataSet::n_classes() const {
    return label_.size();
}

const vector<int> &DataSet::label() const {
    return label_;
}

void DataSet::group_classes() {
    start_.clear();
    count_.clear();
    label_.clear();
    perm_.clear();
    vector<int> dataLabel(y_.size());//temporary labels of all the instances

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
            //real to int conversion is safe, because group_classes only used in classification
            label_.push_back(int(y_[i]));
            count_.push_back(1);
        }
    }

    //logically put instances of the same class consecutively.
    start_.push_back(0);
    for (int i = 1; i < count_.size(); ++i) {
        start_.push_back(start_[i - 1] + count_[i - 1]);
    }
    vector<int> start_copy(start_);
    perm_ = vector<int>(y_.size());//index of each instance in the original array
    for (int i = 0; i < y_.size(); ++i) {
        perm_[start_copy[dataLabel[i]]] = i;
        start_copy[dataLabel[i]]++;
    }
}

size_t DataSet::total_count() const {//return the total number of instances
    return total_count_;
}

size_t DataSet::n_features() const {
    return n_features_;
}

const DataSet::node2d &DataSet::instances() const {//return all the instances
    return instances_;
}

const DataSet::node2d DataSet::instances(int y_i) const {//return instances of a given class
    int si = start_[y_i];
    int ci = count_[y_i];
    node2d one_class_ins;
    for (int i = si; i < si + ci; ++i) {
        one_class_ins.push_back(instances_[perm_[i]]);
    }
    return one_class_ins;
}

const DataSet::node2d DataSet::instances(int y_i, int y_j) const {//return instances of two classes
    node2d two_class_ins;
    node2d i_ins = instances(y_i);
    node2d j_ins = instances(y_j);
    two_class_ins.insert(two_class_ins.end(), i_ins.begin(), i_ins.end());
    two_class_ins.insert(two_class_ins.end(), j_ins.begin(), j_ins.end());
    return two_class_ins;
}

const vector<int> DataSet::original_index() const {//index of each instance in the original array
    return perm_;
}

const vector<int> DataSet::original_index(int y_i) const {//index of each instance in the original array for one class
    int si = start_[y_i];
    int ci = count_[y_i];
    vector<int> one_class_idx;
    for (int i = si; i < si + ci; ++i) {
        one_class_idx.push_back(perm_[i]);
    }
    return one_class_idx;
}

const vector<int>
DataSet::original_index(int y_i, int y_j) const {//index of each instance in the original array for two class
    vector<int> two_class_idx;
    vector<int> i_idx = original_index(y_i);
    vector<int> j_idx = original_index(y_j);
    two_class_idx.insert(two_class_idx.end(), i_idx.begin(), i_idx.end());
    two_class_idx.insert(two_class_idx.end(), j_idx.begin(), j_idx.end());
    return two_class_idx;
}

const vector<real> &DataSet::y() const {
    return y_;
}


