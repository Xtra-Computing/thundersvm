//
// Created by jiashuai on 17-9-20.
//
#include "thundersvm/kernelmatrix.h"
#include "thundersvm/kernel/kernelmatrix_kernel.h"

KernelMatrix::KernelMatrix(const vector<vector<int>> &idx, const vector<vector<real>> &val, size_t n_features,
                           real gamma) : val_(SyncData<real>(0)), col_ind_(SyncData<int>(0)),
                                         row_ptr_(SyncData<int>(0)), self_dot_(SyncData<real>(0)),
                                         diag_(SyncData<real>(0)) {
    m_ = idx.size();
    n_ = n_features;
    this->gamma = gamma;
    vector<real> csr_val;
    vector<int> csr_col_ind;
    vector<int> csr_row_ptr(1, 0);
    vector<real> csr_self_dot;
    for (int i = 0; i < idx.size(); ++i) {
        real self_dot = 0;
        for (int j = 0; j < idx[i].size(); ++j) {
            csr_val.push_back(val[i][j]);
            self_dot += val[i][j] * val[i][j];
            csr_col_ind.push_back(idx[i][j] - 1);//libSVM data format is one-based, convert to zero-based
        }
        csr_row_ptr.push_back(csr_row_ptr.back() + val[i].size());
        csr_self_dot.push_back(self_dot);
    }
    val_.resize(csr_val.size());
    col_ind_.resize(csr_col_ind.size());
    row_ptr_.resize(csr_row_ptr.size());
    self_dot_.resize(m_);
    val_.copy_from(csr_val.data(), val_.count());
    col_ind_.copy_from(csr_col_ind.data(), col_ind_.count());
    row_ptr_.copy_from(csr_row_ptr.data(), row_ptr_.count());
    self_dot_.copy_from(csr_self_dot.data(), self_dot_.count());
    nnz_ = csr_val.size();
    diag_.resize(m_);
    for (int i = 0; i < m_; ++i) {
        diag_.host_data()[i] = 1;//rbf kernel
    }

    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
}

void KernelMatrix::get_rows(const SyncData<int> *idx, SyncData<real> *kernel_rows) const {
    CHECK_EQ(val_.head(), SyncMem::DEVICE);
    CHECK_EQ(row_ptr_.head(), SyncMem::DEVICE);
    CHECK_EQ(col_ind_.head(), SyncMem::DEVICE);
    CHECK_EQ(self_dot_.head(), SyncMem::DEVICE);
    CHECK_EQ(idx->head(), SyncMem::DEVICE);
    CHECK_EQ(kernel_rows->count(), idx->count() * m_) << "kernel_rows memory is too small";

    SyncData<real> data_rows(idx->count() * n_);
    CUDA_CHECK(cudaMemset(data_rows.device_data(), 0, data_rows.size()));
    kernel_get_data_rows << < 1, idx->count() >> >
                                 (val_.device_data(), col_ind_.device_data(), row_ptr_.device_data(), idx->device_data(), data_rows.device_data(), idx->count());
    float one(1);
    float zero(0);
    cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                    m_, idx->count(), n_, nnz_, &one, descr, val_.device_data(), row_ptr_.device_data(),
                    col_ind_.device_data(),
                    data_rows.device_data(), idx->count(), &zero, kernel_rows->device_data(), m_);
    kernel_RBF_kernel << < ((idx->count()) * m_ - 1) / 512 + 1, 512 >> >
                                                                (idx->device_data(), self_dot_.device_data(), kernel_rows->device_data(), idx->count(), m_, gamma);
}

const SyncData<real> *KernelMatrix::diag() const {
    return &this->diag_;
}

KernelMatrix::~KernelMatrix() {
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}
