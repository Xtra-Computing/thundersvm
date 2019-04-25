//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERSVM_DATASET_H
#define THUNDERSVM_DATASET_H

#include "thundersvm.h"
#include "syncarray.h"

/**
 * @brief Dataset reader
 */
class DataSet {
public:
    struct node{
        node(int index, kernel_type value) : index(index), value(value) {}

        int index;
        kernel_type value;
    };

    typedef vector<vector<DataSet::node>> node2d;

    DataSet();

    /**
     * construct a dataset using given instances
     * @param instances given instances
     * @param n_features the number of features of given instances
     * @param y the label of each instances
     */
    DataSet(const DataSet::node2d &instances, int n_features, const vector<float_type> &y);

    ///load dataset from file
    void load_from_file(string file_name);

    ///load dataset from python
    void load_from_python(float *y, char **x, int len);

    ///load from sparse data
    void load_from_sparse(int row_size, float* val, int* row_ptr, int* col_ptr, float* label);

    ///load from dense data
    void load_from_dense(int row_size, int features, float* data, float* label);

    ///group instances in same class
    void group_classes(bool classification = true);

    size_t n_instances() const;

    size_t n_features() const;

    size_t n_classes() const;

    ///the number of instances for each class
    const vector<int> &count() const;

    ///the start position of instances for each class
    const vector<int> &start() const;

    ///mapping logical label (0,1,2,3,...) to real label (maybe 2,4,5,6,...)
    const vector<int> &label() const;

    ///label for each instances, the instances are arranged as they are in file
    const vector<float_type> &y() const;

    const node2d & instances() const;

    ///instances of class \f$y_i\f$
    const node2d instances(int y_i) const;

    ///instances of class \f$y_i\f$ and \f$y_j\f$
    const node2d instances(int y_i, int y_j) const;

    ///mapping instance index (after grouped) to the original index (in file)
    const vector<int> original_index() const;

    const vector<int> original_index(int y_i) const;

    const vector<int> original_index(int y_i, int y_j) const;

	const bool is_zero_based() const;
private:
    vector<float_type> y_;
    node2d instances_;
    size_t total_count_;
    size_t n_features_;
    vector<int> start_; //logical start position of each class
    vector<int> count_; //the number of instances of each class
    vector<int> label_;
    vector<int> perm_;
	bool zero_based = 0; //is zero_based format dataset?
};
#endif //THUNDERSVM_DATASET_H
