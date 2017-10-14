//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/svmparam.h>
#include "thundersvm/kernelmatrix.h"
#include "thundersvm/kernel/kernelmatrix_kernel.h"

KernelMatrix::KernelMatrix(const DataSet::node2d &instances, SvmParam param) {
    n_instances_ = instances.size();
    n_features_ = 0;
    this->param = param;

    //three arrays for csr representation
    vector<real> csr_val;
    vector<int> csr_col_ind;//index of each value of all the instances
    vector<int> csr_row_ptr(1, 0);//the start positions of the instances

    vector<real> csr_self_dot;
    for (int i = 0; i < n_instances_; ++i) {//convert libsvm format to csr format
        real self_dot = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            csr_val.push_back(instances[i][j].value);
            self_dot += instances[i][j].value * instances[i][j].value;
            csr_col_ind.push_back(instances[i][j].index - 1);//libSVM data format is one-based, convert to zero-based
            if (instances[i][j].index > n_features_) n_features_ = instances[i][j].index;
        }
        csr_row_ptr.push_back(csr_row_ptr.back() + instances[i].size());
        csr_self_dot.push_back(self_dot);
    }

    //three arrays (on GPU/CPU) for csr representation
    val_ = new SyncData<real>(csr_val.size());
    col_ind_ = new SyncData<int>(csr_col_ind.size());
    row_ptr_ = new SyncData<int>(csr_row_ptr.size());
    //copy data to the three arrays
    val_->copy_from(csr_val.data(), val_->count());
    col_ind_->copy_from(csr_col_ind.data(), col_ind_->count());
    row_ptr_->copy_from(csr_row_ptr.data(), row_ptr_->count());

    self_dot_ = new SyncData<real>(n_instances_);
    self_dot_->copy_from(csr_self_dot.data(), self_dot_->count());

    nnz_ = csr_val.size();//number of nonzero

    //pre-compute diagonal elements

    diag_ = new SyncData<real>(n_instances_);
    switch (param.kernel_type) {
        case SvmParam::RBF:
            for (int i = 0; i < n_instances_; ++i) {
                diag_->host_data()[i] = 1;//rbf kernel
            }
            break;
        case SvmParam::LINEAR:
            diag_->copy_from(*self_dot_);
            break;
        case SvmParam::POLY:
            diag_->copy_from(*self_dot_);
            SAFE_KERNEL_LAUNCH(kernel_poly_kernel, diag_->device_data(), param.gamma, param.coef0, param.degree,
                               diag_->count());
            break;
        case SvmParam::SIGMOID:
            diag_->copy_from(*self_dot_);
            SAFE_KERNEL_LAUNCH(kernel_sigmoid_kernel, diag_->device_data(), param.gamma, param.coef0, diag_->count());
        default:
            break;
    }

    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
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

void KernelMatrix::get_rows(const SyncData<int> &idx,
                            SyncData<real> &kernel_rows) const {//compute multiple rows of kernel matrix according to idx
    CHECK_GE(kernel_rows.count(), idx.count() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(idx, kernel_rows);
    switch (param.kernel_type) {
        case SvmParam::RBF:
        SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, idx.device_data(), self_dot_->device_data(), kernel_rows.device_data(),
                           idx.count(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
        SAFE_KERNEL_LAUNCH(kernel_poly_kernel, kernel_rows.device_data(), param.gamma, param.coef0, param.degree,
                           kernel_rows.count());
            break;
        case SvmParam::SIGMOID:
        SAFE_KERNEL_LAUNCH(kernel_sigmoid_kernel, kernel_rows.device_data(), param.gamma, param.coef0,
                           kernel_rows.count());
            break;
    }
}

void KernelMatrix::get_rows(const DataSet::node2d &instances,
                            SyncData<real> &kernel_rows) const {//compute the whole (sub-) kernel matrix of the given instances.
    CHECK_GE(kernel_rows.count(), instances.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(instances, kernel_rows);

    //compute self dot
    //TODO use thrust
    SyncData<real> self_dot(instances.size());
    for (int i = 0; i < instances.size(); ++i) {
        real sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            sum += instances[i][j].value * instances[i][j].value;
        }
        self_dot[i] = sum;
    }
    switch (param.kernel_type) {
        case SvmParam::RBF:
        SAFE_KERNEL_LAUNCH(kernel_RBF_kernel, self_dot.device_data(), this->self_dot_->device_data(),
                           kernel_rows.device_data(), instances.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
        SAFE_KERNEL_LAUNCH(kernel_poly_kernel, kernel_rows.device_data(), param.gamma, param.coef0, param.degree,
                           kernel_rows.count());
            break;
        case SvmParam::SIGMOID:
        SAFE_KERNEL_LAUNCH(kernel_sigmoid_kernel, kernel_rows.device_data(), param.gamma, param.coef0,
                           kernel_rows.count());
            break;
    }
}

const SyncData<real> &KernelMatrix::diag() const {
    return *this->diag_;
}


void KernelMatrix::dns_csr_mul(const SyncData<real> &dense_mat, int n_rows, SyncData<real> &result) const {
    CHECK_EQ(dense_mat.count(), n_rows * n_features_) << "dense matrix features doesn't match";
    float one(1);
    float zero(0);
    cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                    n_instances_, n_rows, n_features_, nnz_, &one, descr, val_->device_data(), row_ptr_->device_data(),
                    col_ind_->device_data(),
                    dense_mat.device_data(), n_rows, &zero, result.device_data(), n_instances_);
    //cusparseScsrmm return row-major matrix, so no transpose is needed
}

void KernelMatrix::get_dot_product(const SyncData<int> &idx, SyncData<real> &dot_product) const {
    SyncData<real> data_rows(idx.count() * n_features_);
    data_rows.mem_set(0);
    SAFE_KERNEL_LAUNCH(kernel_get_working_set_ins, val_->device_data(), col_ind_->device_data(),
                       row_ptr_->device_data(),
                       idx.device_data(), data_rows.device_data(), idx.count());
    dns_csr_mul(data_rows, idx.count(), dot_product);
}

void KernelMatrix::get_dot_product(const DataSet::node2d &instances, SyncData<real> &dot_product) const {
    SyncData<real> dense_ins(instances.size() * n_features_);
    dense_ins.mem_set(0);
    for (int i = 0; i < instances.size(); ++i) {
        real sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            CHECK_LE(instances[i][j].index, n_features_)
                << "the number of features in testing set is larger than training set";
            dense_ins[(instances[i][j].index - 1) * instances.size() + i] = instances[i][j].value;
            sum += instances[i][j].value * instances[i][j].value;
        }
    }
    dns_csr_mul(dense_ins, instances.size(), dot_product);
}


