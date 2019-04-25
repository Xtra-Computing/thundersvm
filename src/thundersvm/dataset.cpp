//
// Created by jiashuai on 17-9-17.
//
#include "thundersvm/dataset.h"
#include <omp.h>

using std::fstream;
using std::stringstream;

DataSet::DataSet() : total_count_(0), n_features_(0) {}

DataSet::DataSet(const DataSet::node2d &instances, int n_features, const vector<float_type> &y) :
        instances_(instances), n_features_(n_features), y_(y), total_count_(instances_.size()) {}

inline char *findlastline(char *ptr, char *begin) {
    while (ptr != begin && *ptr != '\n') --ptr;
    return ptr;
}

void DataSet::load_from_file(string file_name) {
    LOG(INFO)<<"loading dataset from file \""<<file_name<<"\"";
    y_.clear();
    instances_.clear();
    total_count_ = 0;
    n_features_ = 0;
    std::ifstream ifs(file_name, std::ifstream::binary);
    if(!ifs.is_open()){
		LOG(INFO)<<"file "<<file_name<<" not found";
		exit(1);
	}
	//CHECK(ifs.is_open()) << "file " << file_name << " not found";

    int buffer_size = 16 << 20; //16MB
	char *buffer = (char *)malloc(buffer_size);
    const int nthread = omp_get_max_threads();
    while (ifs) {
        char *head = buffer;
        ifs.read(buffer, buffer_size);
        size_t size = ifs.gcount();
        vector<vector<float_type>> y_thread(nthread);
        vector<node2d> instances_thread(nthread);

        vector<int> local_feature(nthread, 0);
#pragma omp parallel num_threads(nthread)
        {
            //get working area of this thread
            int tid = omp_get_thread_num();
            size_t nstep = (size + nthread - 1) / nthread;
            size_t sbegin = min(tid * nstep, size - 1);
            size_t send = min((tid + 1) * nstep, size - 1);
            char *pbegin = findlastline(head + sbegin, head);
            char *pend = findlastline(head + send, head);

            //move stream start position to the end of last line
            if (tid == nthread - 1) ifs.seekg(pend - head - send, std::ios_base::cur);
            //read instances line by line
            char *lbegin = pbegin;
            char *lend = lbegin;
            while (lend != pend) {
                //get one line
                lend = lbegin + 1;
                while (lend != pend && *lend != '\n') {
                    ++lend;
                }
                string line(lbegin, lend);
                stringstream ss(line);

                //read label of an instance
                y_thread[tid].emplace_back();
                ss >> y_thread[tid].back();

                //read features of an instance
                instances_thread[tid].emplace_back();
                string tuple;
                while (ss >> tuple) {
                    int i;
                    float v;
                    CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2) << "read error, using [index]:[value] format";
                    instances_thread[tid].back().emplace_back(i, v);
					if(i == 0 && zero_based == 0) zero_based = 1;
                    if (i > local_feature[tid]) local_feature[tid] = i;
                };

                //read next instance
                lbegin = lend;
            }
        }
        for (int i = 0; i < nthread; i++) {
            if (local_feature[i] > n_features_)
                n_features_ = local_feature[i];
            total_count_ += instances_thread[i].size();
        }
        for (int i = 0; i < nthread; i++) {
            this->y_.insert(y_.end(), y_thread[i].begin(), y_thread[i].end());
            this->instances_.insert(instances_.end(), instances_thread[i].begin(), instances_thread[i].end());
        }
    }
    free(buffer);
    LOG(INFO)<<"#instances = "<<this->n_instances()<<", #features = "<<this->n_features();
}

void DataSet::load_from_python(float *y, char **x, int len) {
    y_.clear();
    instances_.clear();
    total_count_ = 0;
    n_features_ = 0;
    for (int i = 0; i < len; i++) {
        int ind;
        float v;
        string line = x[i];
        stringstream ss(line);
        y_.push_back(y[i]);
        instances_.emplace_back();
        string tuple;
        while (ss >> tuple) {
            CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &ind, &v), 2) << "read error, using [index]:[value] format";
            instances_[total_count_].emplace_back(ind, v);
            if (ind > n_features_) n_features_ = ind;
        };
        total_count_++;
    }
}

void DataSet::load_from_sparse(int row_size, float* val, int* row_ptr, int* col_ptr, float* label) {
    y_.clear();
    instances_.clear();
    total_count_ = 0;
    n_features_ = 0;
    for(int i = 0; i < row_size; i++){
        int ind;
        float  v;
        if(label != NULL)
            y_.push_back(label[i]);
        instances_.emplace_back();
        for(int i = row_ptr[total_count_]; i < row_ptr[total_count_ + 1]; i++){
            ind = col_ptr[i];
			ind++;			//convert to one-based format
            v = val[i];
            instances_[total_count_].emplace_back(ind, v);
            if(ind > n_features_) n_features_ = ind;
        }
        total_count_++;

    }
//    n_features_++;
    LOG(INFO)<<"#instances = "<<this->n_instances()<<", #features = "<<this->n_features();

}

void DataSet::load_from_dense(int row_size, int features, float* data, float* label){
    y_.clear();
    instances_.clear();
    total_count_ = 0;
    n_features_ = 0;
    int off = 0;
    for(int i = 0; i < row_size; i++){
        int ind;
        float v;
        if(label != NULL)
            y_.push_back(label[i]);
        instances_.emplace_back();
        for(int j = 1; j <= features; j++){
            ind = j;
            v = data[off];
            off++;
            instances_[total_count_].emplace_back(ind, v);
        }
        total_count_++;
    }
    n_features_ = features;
    LOG(INFO)<<"#instances = "<<this->n_instances()<<", #features = "<<this->n_features();
}

const vector<int> &DataSet::count() const {//return the number of instances of each class
    return count_;
}

const vector<int> &DataSet::start() const {
    return start_;
}

size_t DataSet::n_classes() const {
    return start_.size();
}

const vector<int> &DataSet::label() const {
    return label_;
}

void DataSet::group_classes(bool classification) {
    if (classification) {
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
    } else {
        for (int i = 0; i < instances_.size(); ++i) {
            perm_.push_back(i);
        }
        start_.push_back(0);
        count_.push_back(instances_.size());
    }
}

size_t DataSet::n_instances() const {//return the total number of instances
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
    return vector<int>(perm_.begin() + start_[y_i], perm_.begin() + start_[y_i] + count_[y_i]);
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

const vector<float_type> &DataSet::y() const {
    return y_;
}

const bool DataSet::is_zero_based() const{
	return zero_based;
}
