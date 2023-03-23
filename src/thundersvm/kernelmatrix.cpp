//
// Created by jiashuai on 17-9-20.
//
#include <thundersvm/svmparam.h>
#include "thundersvm/kernelmatrix.h"
#include "thundersvm/kernel/kernelmatrix_kernel.h"
#include <cusparse.h>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()

using namespace svm_kernel;
extern long long time1;
extern long long time2;
extern long long time3;
extern long long time4;
KernelMatrix::KernelMatrix(const DataSet::node2d &instances, SvmParam param) {
    n_instances_ = instances.size();
    n_features_ = 0;
    this->param = param;

    //three arrays for csr representation
    vector<kernel_type> csr_val;
    vector<int> csr_col_ind;//index of each value of all the instances
    vector<int> csr_row_ptr(1, 0);//the start positions of the instances

    vector<kernel_type> csr_self_dot;
    for (int i = 0; i < n_instances_; ++i) {//convert libsvm format to csr format
        float_type self_dot = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            csr_val.push_back(instances[i][j].value);
            self_dot += instances[i][j].value * instances[i][j].value;
            csr_col_ind.push_back(instances[i][j].index);//libSVM data format is one-based, convert to zero-based
            if (instances[i][j].index > n_features_) n_features_ = instances[i][j].index;
        }
        csr_row_ptr.push_back(csr_row_ptr.back() + instances[i].size());
        csr_self_dot.push_back(self_dot);
    }
    n_features_++;

    //three arrays (on GPU/CPU) for csr representation
    val_.resize(csr_val.size());
    col_ind_.resize(csr_col_ind.size());
    row_ptr_.resize(csr_row_ptr.size());
    //copy data to the three arrays
    val_.copy_from(csr_val.data(), val_.size());
    col_ind_.copy_from(csr_col_ind.data(), col_ind_.size());
    row_ptr_.copy_from(csr_row_ptr.data(), row_ptr_.size());

    self_dot_.resize(n_instances_);
    self_dot_.copy_from(csr_self_dot.data(), self_dot_.size());

    nnz_ = csr_val.size();//number of nonzero
    //pre-compute diagonal elements

    diag_.resize(n_instances_);
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            for (int i = 0; i < n_instances_; ++i) {
                diag_.host_data()[i] = 1;//rbf kernel
            }
            break;
        case SvmParam::LINEAR:
            diag_.copy_from(self_dot_);
            break;
        case SvmParam::POLY:
            diag_.copy_from(self_dot_);
            poly_kernel(diag_, param.gamma, param.coef0, param.degree, diag_.size());
            break;
        case SvmParam::SIGMOID:
            diag_.copy_from(self_dot_);
            sigmoid_kernel(diag_, param.gamma, param.coef0, diag_.size());
        default:
            break;
    }

}

void KernelMatrix::get_rows(const SyncArray<int> &idx,
                            SyncArray<kernel_type> &kernel_rows) const {//compute multiple rows of kernel matrix according to idx
    CHECK_GE(kernel_rows.size(), idx.size() * n_instances_) << "kernel_rows memory is too small";


    //check values 
    //LOG(INFO)<<idx.size()<<" "<<n_instances_<<" "<<n_features_<<" "<<nnz_;
#ifdef USE_CUDA
    get_dot_product_dns_csr(idx, kernel_rows);
    //get_dot_product_csr_csr_cuda(idx, kernel_rows);
#else
	if(n_features_ < 1000000)
		get_dot_product_dns_csr(idx, kernel_rows);
	else
		get_dot_product_csr_csr(idx, kernel_rows);
//    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            RBF_kernel(idx, self_dot_, kernel_rows, idx.size(), n_instances_, param.gamma);
			break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
}

void KernelMatrix::get_dot_product_dns_bsr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product,
                                            SyncArray<kernel_type> &bsr_val,SyncArray<int> &bsr_offset,SyncArray<int> &bsr_col) const{
    SyncArray<kernel_type> data_rows(idx.size() * n_features_);
    data_rows.mem_set(0);
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size(), n_features_);

    svm_kernel::bsr_dns_mul(n_instances_, idx.size(), n_features_, data_rows, bsr_val,
                     bsr_offset, bsr_col,
                     dot_product);

}
//bsr dns mul
void KernelMatrix::get_rows_bsr(const SyncArray<int> &idx, SyncArray<kernel_type> &kernel_rows,
                                SyncArray<kernel_type> &bsr_val,SyncArray<int> &bsr_offset,SyncArray<int> &bsr_col) const{

    CHECK_GE(kernel_rows.size(), idx.size() * n_instances_) << "kernel_rows memory is too small";


    //check values 
    //LOG(INFO)<<idx.size()<<" "<<n_instances_<<" "<<n_features_<<" "<<nnz_;
#ifdef USE_CUDA
    get_dot_product_dns_bsr(idx, kernel_rows,bsr_val,bsr_offset,bsr_col);
    //get_dot_product_csr_csr_cuda(idx, kernel_rows);
#else
    LOG(INFO)<<"no implementing !!!!!!!!!!!!!";
//    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            RBF_kernel(idx, self_dot_, kernel_rows, idx.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }

}


//new method 

void KernelMatrix::get_sparse_dense_rows(const SyncArray<int> &idx, SparseData &sparse,DenseData &dense,SyncArray<kernel_type> &kernel_rows) const {
     CHECK_GE(kernel_rows.size(), idx.size() * n_instances_) << "kernel_rows memory is too small";


    //check values 
    // LOG(INFO)<<" check sparse matrix shape is "<<sparse.row<<" "<<sparse.col<<" "<<sparse.is_use;
#ifdef USE_CUDA
    
    get_dot_product_dns_csr_dns_dns(idx, sparse,dense,kernel_rows);
    
#else
    LOG(INFO)<<"no implementing !!!!!!!!!!!!!";
//    get_dot_product_dns_dns(idx, kernel_rows);
#endif
    TDEF(kernel)
    TSTART(kernel)
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            RBF_kernel(idx, self_dot_, kernel_rows, idx.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
    cudaDeviceSynchronize();
    TEND(kernel)
    time3+=TINT(kernel);
}

void KernelMatrix::get_dot_product_dns_csr_dns_dns(const SyncArray<int> &idx,SparseData &sparse,DenseData &dense,SyncArray<kernel_type> &dot_product) const{
    TDEF(dot_product)
    TSTART(dot_product)

    //首先分别得到用来和稀疏矩阵和稠密矩阵相乘的两个稠密矩阵
    SyncArray<kernel_type> sparse_data_rows(idx.size() * sparse.col);
    sparse_data_rows.mem_set(0);

    //for sparse
    TDEF(sparse)
    TSTART(sparse)
    get_working_set_ins(sparse.val_, sparse.col_ind_, sparse.row_ptr_, idx, sparse_data_rows, idx.size(), sparse.col); //col-major
    dns_csr_mul_part(sparse_data_rows, idx.size(), sparse,dot_product);
    // get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size(), n_features_);
    
    cudaDeviceSynchronize();
    TEND(sparse)
    time1+=TINT(sparse);
    //for dense 先简单的cpu实现
    const int *data_row_idx_data = idx.host_data();
    
     
    if(dense.is_use){
         SyncArray<kernel_type> dense_data_rows(idx.size() * dense.col);
        dense_data_rows.mem_set(0);
        kernel_type* h_dense_data_rows = dense_data_rows.host_data();

        kernel_type* h_dense = dense.val.host_data();
        
        TDEF(dense)
        TSTART(dense)
        #pragma omp parallel for
        for (int i=0;i<idx.size();i++){
            int row = data_row_idx_data[i];
            for (int j=0;j<dense.col;j++){
                h_dense_data_rows[i*dense.col+j] = h_dense[row*dense.col+j];
            }
        }
        
        //计算得到结果
        kernel_type beta = 1.0;
        dns_dns_mul_part(dense_data_rows,idx.size(),dense,dot_product,beta);

        cudaDeviceSynchronize();
        TEND(dense)
        time2+=TINT(dense);

        TEND(dot_product)
        time4+=TINT(dot_product);

    }

}

void KernelMatrix::dns_csr_mul_part(const SyncArray<kernel_type> &dense_mat, int n_rows, SparseData &sparse,SyncArray<kernel_type> &result) const{
    CHECK_EQ(dense_mat.size(), n_rows * sparse.col) << "dense matrix features doesn't match";

    svm_kernel::dns_csr_mul(n_instances_, n_rows, sparse.col, dense_mat, sparse.val_, sparse.row_ptr_, sparse.col_ind_, sparse.val_.size(), result);
}

void KernelMatrix::dns_dns_mul_part(const SyncArray<kernel_type> &dense_mat, int n_rows,DenseData &dense,SyncArray<kernel_type> &result,kernel_type beta) const{

    svm_kernel::dns_dns_mul(n_instances_, n_rows, dense.col, dense.val,dense_mat,beta,result);
}






void KernelMatrix::get_rows(const DataSet::node2d &instances,
                            SyncArray<kernel_type> &kernel_rows) const {//compute the whole (sub-) kernel matrix of the given instances.
    CHECK_GE(kernel_rows.size(), instances.size() * n_instances_) << "kernel_rows memory is too small";
    get_dot_product(instances, kernel_rows);

    //compute self dot
    //TODO use thrust
    SyncArray<kernel_type> self_dot(instances.size());
    kernel_type *self_dot_data = self_dot.host_data();
    for (int i = 0; i < instances.size(); ++i) {
        kernel_type sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            sum += instances[i][j].value * instances[i][j].value;
        }
        self_dot_data[i] = sum;
    }
    switch (param.kernel_type) {
        case SvmParam::RBF:
        case SvmParam::PRECOMPUTED://precomputed uses rbf as default
            RBF_kernel(self_dot, self_dot_, kernel_rows, instances.size(), n_instances_, param.gamma);
            break;
        case SvmParam::LINEAR:
            //do nothing
            break;
        case SvmParam::POLY:
            poly_kernel(kernel_rows, param.gamma, param.coef0, param.degree, kernel_rows.size());
            break;
        case SvmParam::SIGMOID:
            sigmoid_kernel(kernel_rows, param.gamma, param.coef0, kernel_rows.size());
            break;
    }
}

const SyncArray<kernel_type> &KernelMatrix::diag() const {
    return this->diag_;
}


void
KernelMatrix::dns_csr_mul(const SyncArray<kernel_type> &dense_mat, int n_rows, SyncArray<kernel_type> &result) const {
    CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
    svm_kernel::dns_csr_mul(n_instances_, n_rows, n_features_, dense_mat, val_, row_ptr_, col_ind_, nnz_, result);
}

#ifndef USE_CUDA
void
KernelMatrix::csr_csr_mul(const SyncArray<kernel_type> &ws_val, int n_rows, const SyncArray<int> &ws_col_ind,
                          const SyncArray<int> &ws_row_ptr, SyncArray<kernel_type> &result) const {
    svm_kernel::csr_csr_mul(n_instances_, n_rows, n_features_, ws_val, ws_col_ind, ws_row_ptr,
                            val_, row_ptr_, col_ind_, nnz_, ws_val.size(), result);
}

void
KernelMatrix::dns_dns_mul(const SyncArray<kernel_type> &dense_mat, int n_rows,
                          const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result) const {
    CHECK_EQ(dense_mat.size(), n_rows * n_features_) << "dense matrix features doesn't match";
    svm_kernel::dns_dns_mul(n_instances_, n_rows, n_features_, dense_mat, origin_dense, result);
}
#endif
void KernelMatrix::get_dot_product_dns_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const {
    SyncArray<kernel_type> data_rows(idx.size() * n_features_);

    TDEF(sparse)
    TSTART(sparse)
    data_rows.mem_set(0);
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size(), n_features_);
    dns_csr_mul(data_rows, idx.size(), dot_product);

    cudaDeviceSynchronize();
    TEND(sparse)
    time1+=TINT(sparse);
}

void KernelMatrix::get_dot_product(const DataSet::node2d &instances, SyncArray<kernel_type> &dot_product) const {
    SyncArray<kernel_type> dense_ins(instances.size() * n_features_);
    dense_ins.mem_set(0);
    kernel_type *dense_ins_data = dense_ins.host_data();
    for (int i = 0; i < instances.size(); ++i) {
        kernel_type sum = 0;
        for (int j = 0; j < instances[i].size(); ++j) {
            if (instances[i][j].index < n_features_) {
                //col major for cuSPARSE, row major for Eigen
#ifdef USE_CUDA
                dense_ins_data[instances[i][j].index * instances.size() + i] = instances[i][j].value;
#else
                dense_ins_data[i * n_features_ + instances[i][j].index] = instances[i][j].value;
#endif
                sum += instances[i][j].value * instances[i][j].value;
            } else {
//                LOG(WARNING)<<"the number of features in testing set is larger than training set";
            }
        }
    }
    dns_csr_mul(dense_ins, instances.size(), dot_product);
}
#ifndef USE_CUDA
void KernelMatrix::get_dot_product_csr_csr(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const {
    SyncArray<kernel_type> ws_val;
    SyncArray<int> ws_col_ind;
    SyncArray<int> ws_row_ptr;
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, ws_val, ws_col_ind, ws_row_ptr, idx.size());
    csr_csr_mul(ws_val, idx.size(), ws_col_ind, ws_row_ptr, dot_product);
}

void KernelMatrix::get_dot_product_dns_dns(const SyncArray<int> &idx, SyncArray<kernel_type> &dot_product) const {
    SyncArray<kernel_type> data_rows(idx.size() * n_features_);
    data_rows.mem_set(0);
    SyncArray<kernel_type> origin_dense(n_instances_ * n_features());
    origin_dense.mem_set(0);
    SyncArray<int> origin_idx(n_instances_);
    int *origin_idx_data = origin_idx.host_data();
    for (int i = 0; i < n_instances_; ++i) {
        origin_idx_data[i] = i;
    }
    get_working_set_ins(val_, col_ind_, row_ptr_, idx, data_rows, idx.size(), n_features_);
    get_working_set_ins(val_, col_ind_, row_ptr_, origin_idx, origin_dense, origin_idx.size(), n_features_);
    dns_dns_mul(data_rows, idx.size(), origin_dense, dot_product);
}
#endif
