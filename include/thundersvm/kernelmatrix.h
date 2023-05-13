//
// Created by jiashuai on 17-9-19.
//

#ifndef THUNDERSVM_KERNELMATRIX_H
#define THUNDERSVM_KERNELMATRIX_H

#include "thundersvm.h"
#include "syncarray.h"
#include "dataset.h"
#include "svmparam.h"


//csr to csr part and dense part
struct SparseData{
    SyncArray<kernel_type> val_;
    SyncArray<int> col_ind_;
    SyncArray<int> row_ptr_;
    // int* table;
    int row;
    int col;
    bool is_use = false;
};


struct DenseData{
    SyncArray<kernel_type> val;
    int row = 0;
    int col = 0;
    // int* Ttable;
    // int* Ftable;
    bool is_use = false;
};


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
    void get_rows(const SyncArray<int> &idx, SyncArray<kernel_type> &kernel_rows) const;

    /**
     * return kernel values of each given instance and each instance stored in KernelMatrix
     * @param [in] instances the given instances
     * @param [out] kernel_rows
     */
    void get_rows(const DataSet::node2d &instances, SyncArray<kernel_type> &kernel_rows) const;

    ///return the diagonal elements of kernel matrix
    const SyncArray<kernel_type> &diag() const;

    ///the number of instances in KernelMatrix
    size_t n_instances() const { return n_instances_; };

    ///the maximum number of features of instances
    size_t n_features() const { return n_features_; }

    ///the number of non-zero features of all instances
    size_t nnz() const {return nnz_;};//number of nonzero
private:
    KernelMatrix &operator=(const KernelMatrix &) const;

    KernelMatrix(const KernelMatrix &);

    SyncArray<kernel_type> val_;
    SyncArray<int> col_ind_;
    SyncArray<int> row_ptr_;
    SyncArray<kernel_type> diag_;
    SyncArray<kernel_type> self_dot_;

    //get sparse and dense
    SparseData sparse_mat_;
    DenseData dense_mat_;
    SparseData bsr_sparse;
    SparseData csr_sparse;


    size_t nnz_;
    size_t n_instances_;
    size_t n_features_;
    SvmParam param;
    void dns_csr_mul(const SyncArray<kernel_type> &dense_mat, int n_rows, SyncArray<kernel_type> &result) const;
#ifndef USE_CUDA
    void csr_csr_mul(const SyncArray<kernel_type> &ws_val, int n_rows, const SyncArray<int> &ws_col_ind,
                              const SyncArray<int> &ws_row_ptr, SyncArray<kernel_type> &result) const;
    void dns_dns_mul(const SyncArray<kernel_type> &dense_mat, int n_rows,
                              const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result) const;
#endif
    void get_dot_product_dns_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;

    void get_dot_product_csr_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;

    void get_dot_product_dns_dns(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;

    void get_dot_product(const DataSet::node2d &instances, SyncArray<kernel_type> &dot_product) const;

    void get_dot_product_sparse(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;

    //function for matrix partitioning
    void get_dot_product_dns_csr_dns_dns(const SyncArray<int> &idx,const SparseData &sparse,const DenseData &dense,SyncArray<kernel_type> &dot_product) const;
    void dns_csr_mul_part(const SyncArray<kernel_type> &dense_mat, int n_rows,const SparseData &sparse,SyncArray<kernel_type> &result) const;
    void dns_dns_mul_part(const SyncArray<kernel_type> &dense_mat, int n_rows,const DenseData &dense,SyncArray<kernel_type> &result,kernel_type beta) const;
};
#endif //THUNDERSVM_KERNELMATRIX_H
