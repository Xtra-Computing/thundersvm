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
    int* Ttable;
    // int* Ftable;
    bool is_use = false;
};


struct Node{
    int num;
    int x; //for col
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

        //partition the original matrix to two parts and get their dense representation 
        void get_sparse_dense_rows(const SyncArray<int> &idx, SparseData &sparse,DenseData &dense,SyncArray<kernel_type> &kernel_rows) const;

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

        //get csr 
        const kernel_type* get_val_host() const {return val_.host_data();}
        const int* get_col_host() const {return col_ind_.host_data();}
        const int* get_row_host() const {return row_ptr_.host_data();}

        //bsr
        void get_bsr(int blockSize,SyncArray<kernel_type> &bsr_val,SyncArray<int> &bsr_offset,SyncArray<int> &bsr_col) const;
        void get_rows_bsr(const SyncArray<int> &idx, SyncArray<kernel_type> &kernel_rows,
                                        SyncArray<kernel_type> &bsr_val,SyncArray<int> &bsr_offset,SyncArray<int> &bsr_col) const;
    private:
        KernelMatrix &operator=(const KernelMatrix &) const;

        KernelMatrix(const KernelMatrix &);

        SyncArray<kernel_type> val_;
        SyncArray<int> col_ind_;
        SyncArray<int> row_ptr_;
        SyncArray<kernel_type> diag_;
        SyncArray<kernel_type> self_dot_;
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
        
        //new
        void get_dot_product_dns_csr_dns_dns(const SyncArray<int> &idx,SparseData &sparse,DenseData &dense,SyncArray<kernel_type> &dot_product) const;
        void dns_csr_mul_part(const SyncArray<kernel_type> &dense_mat, int n_rows,SparseData &sparse,SyncArray<kernel_type> &result) const;
        void dns_dns_mul_part(const SyncArray<kernel_type> &dense_mat, int n_rows,DenseData &dense,SyncArray<kernel_type> &result,kernel_type beta) const;
        void get_dot_product_csr_csr_cuda(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const;
        void get_dot_product_dns_bsr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product,
                                                    SyncArray<kernel_type> &bsr_val,SyncArray<int> &bsr_offset,SyncArray<int> &bsr_col) const;
};
#endif //THUNDERSVM_KERNELMATRIX_H
