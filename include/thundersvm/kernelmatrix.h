//
// Created by jiashuai on 17-9-19.
//

#ifndef THUNDERSVM_KERNELMATRIX_H
#define THUNDERSVM_KERNELMATRIX_H

#include "thundersvm.h"
#include "syncdata.h"
#include "dataset.h"
#include "svmparam.h"

/**
 * @brief The management class of kernel values.
 */
class KernelMatrix{
public:
    /**
     * Create KernelMatrix with given instances (training data or support vectors).
     * @param instances the instances, either are training instances for training, or are support vectors for prediction.
     * @param param kernel_type in parm is used
     */
    explicit KernelMatrix(const DataSet::node2d &instances, SvmParam param);

    /**
     * return specific rows in kernel matrix
     * @param [in] idx the indices of the rows
     * @param [out] kernel_rows
     */
    void get_rows(const SyncArray<int> &idx, SyncArray<float_type> &kernel_rows) const;

    /**
     * return kernel values of each given instance and each instance stored in KernelMatrix
     * @param [in] instances the given instances
     * @param [out] kernel_rows
     */
    void get_rows(const DataSet::node2d &instances, SyncArray<float_type> &kernel_rows) const;

    ///return the diagonal elements of kernel matrix
    const SyncArray<float_type> &diag() const;

    ///the number of instances in KernelMatrix
    size_t n_instances() const { return n_instances_; };

    ///the maximum number of features of instances
    size_t n_features() const { return n_features_; }

    ///the number of non-zero features of all instances
    size_t nnz() const {return nnz_;};//number of nonzero
private:
    KernelMatrix &operator=(const KernelMatrix &) const;

    KernelMatrix(const KernelMatrix &);

    SyncArray<float_type> val_;
    SyncArray<int> col_ind_;
    SyncArray<int> row_ptr_;
    SyncArray<float_type> diag_;
    SyncArray<float_type> self_dot_;
    size_t nnz_;
    size_t n_instances_;
    size_t n_features_;
    SvmParam param;

    void dns_csr_mul(const SyncArray<float_type> &dense_mat, int n_rows, SyncArray<float_type> &result) const;

    void get_dot_product(const SyncArray<int> &idx, SyncArray<float_type> &dot_product) const;

    void get_dot_product(const DataSet::node2d &instances, SyncArray<float_type> &dot_product) const;
};
#endif //THUNDERSVM_KERNELMATRIX_H
