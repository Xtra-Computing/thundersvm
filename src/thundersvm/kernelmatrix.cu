//
// Created by jiashuai on 17-9-20.
//
#include "thundersvm/kernelmatrix.h"
#include "thundersvm/kernel/kernelmatrix_kernel.h"

KernelMatrix::KernelMatrix(const DataSet::node2d &instances, size_t n_features, real gamma) {
    m_ = instances.size();
    n_ = n_features;
    this->gamma = gamma;
    vector<real> csr_val;
    vector<int> csr_col_ind;
    vector<int> csr_row_ptr(1, 0);
    vector<real> csr_self_dot;
    for (int i = 0; i < m_; ++i) {
        real self_dot = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            csr_val.push_back(instances[i][j].value);
            self_dot += instances[i][j].value * instances[i][j].value;
            csr_col_ind.push_back(instances[i][j].index - 1);//libSVM data format is one-based, convert to zero-based
        }
        csr_row_ptr.push_back(csr_row_ptr.back() + instances[i].size());
        csr_self_dot.push_back(self_dot);
    }
    val_ = new SyncData<real>(csr_val.size());
    col_ind_ = new SyncData<int>(csr_col_ind.size());
    row_ptr_ = new SyncData<int>(csr_row_ptr.size());
    self_dot_ = new SyncData<real>(m_);
    val_->copy_from(csr_val.data(), val_->count());
    col_ind_->copy_from(csr_col_ind.data(), col_ind_->count());
    row_ptr_->copy_from(csr_row_ptr.data(), row_ptr_->count());
    self_dot_->copy_from(csr_self_dot.data(), self_dot_->count());
    nnz_ = csr_val.size();
    diag_ = new SyncData<real>(m_);
    for (int i = 0; i < m_; ++i) {
        diag_->host_data()[i] = 1;//rbf kernel
    }

    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
}

void KernelMatrix::get_rows(const SyncData<int> &idx, SyncData<real> &kernel_rows) const {
    CHECK_EQ(kernel_rows.count(), idx.count() * m_) << "kernel_rows memory is too small";

    SyncData<real> data_rows(idx.count() * n_);
    data_rows.mem_set(0);
    SAFE_KERNEL_LAUNCH(kernel_get_data_rows, val_->device_data(), col_ind_->device_data(), row_ptr_->device_data(),
                       idx.device_data(), data_rows.device_data(), idx.count());
    dns_csr_mul(data_rows, idx.count(), kernel_rows);
    //cusparseScsrmm return row-major matrix, so no transpose is needed
    SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, idx.device_data(), self_dot_->device_data(), kernel_rows.device_data(),
                       idx.count(), m_, gamma);
}

void KernelMatrix::get_rows(const DataSet::node2d &instances, SyncData<real> &kernel_rows) const {
    CHECK_EQ(kernel_rows.count(), instances.size() * m_) << "kernel_rows memory is too small";
    SyncData<real> self_dot(instances.size());
    SyncData<real> dense_ins(instances.size() * n_);
    for (int i = 0; i < instances.size(); ++i) {
        real sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            dense_ins[(instances[i][j].index - 1) * instances.size() + i] = instances[i][j].value;
            sum += instances[i][j].value * instances[i][j].value;
        }
        self_dot[i] = sum;
    }
    dns_csr_mul(dense_ins, instances.size(), kernel_rows);
    SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, self_dot.device_data(), self_dot_->device_data(), kernel_rows.device_data(),
                       instances.size(), m_, gamma);
}

const SyncData<real> *KernelMatrix::diag() const {
    return this->diag_;
}

KernelMatrix::~KernelMatrix() {
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
    delete val_;
    delete col_ind_;
    delete row_ptr_;
    delete self_dot_;
    delete diag_;
}

void KernelMatrix::dns_csr_mul(const SyncData<real> &dense_mat, int n_rows, SyncData<real> &result) const {
    CHECK_EQ(dense_mat.count(), n_rows * n_) << "dense matrix features doesn't match";
    float one(1);
    float zero(0);
    cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                    m_, n_rows, n_, nnz_, &one, descr, val_->device_data(), row_ptr_->device_data(),
                    col_ind_->device_data(),
                    dense_mat.device_data(), n_rows, &zero, result.device_data(), m_);
}

