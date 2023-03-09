//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/csmosolver.h>
#include <thundersvm/kernel/smo_kernel.h>
#include <climits>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()

using namespace svm_kernel;
long long time1 = 0;
long long time2 = 0;
long long time3 = 0;


// //csr to csr part and dense part
// struct SparseData{
//     SyncArray<kernel_type> val_;
//     SyncArray<int> col_ind_;
//     SyncArray<int> row_ptr_;
//     // int* table;
//     int row;
//     int col;
//     bool is_use = false;
// };


// struct DenseData{
//     SyncArray<kernel_type> val;
//     int row;
//     int col;
//     int* Ttable;
//     // int* Ftable;
//     bool is_use = false;
// };


// struct Node{
//     int num;
//     int x; //for col
// };
void CSR_DenseCSR(const KernelMatrix &k_mat, DenseData &dense,SparseData &sparse){
    const long long m = k_mat.n_instances();

    const long long n = k_mat.n_features();


    const kernel_type * csr_val = k_mat.get_val_host();
    const int * csr_row_ptr = k_mat.get_row_host();
    const int * csr_col_ind = k_mat.get_col_host();



    /*计算每一列的稀疏程度
    然后根据阈值划分为稀疏矩阵和稠密矩阵*/

    Node* col_num = new Node[n+10];

    for (int i=0;i<n;i++) {
        col_num[i].num=0;
        col_num[i].x=i;
    }


    for (int i=0;i<m;i++){
        int csr_row_begin = csr_row_ptr[i];
        int csr_row_end = csr_row_ptr[i+1];
        for (int j=csr_row_begin;j<csr_row_end;j++){
            col_num[csr_col_ind[j]].num++;
        }
    }

    //sort spare to dense 

    std::sort(col_num,col_num+n,[=](Node a,Node b){
        if (a.num<b.num) return true;
        if (a.num==b.num && a.x < b.x) return true;
        return false;
    });


    //start partition

       
    long long row_sum_num=0; //for calculating ratio

    bool* densefg = new bool[n];
    for (int i=0;i<n;i++) densefg[i]=true;


    int densecolnum=n;

    long long total_csr=0;

    long long total_dense=m*n;

    double ratio;
    for (int i = 0; i < n; ++i)
    {
        total_csr+=col_num[i].num;
        total_dense-=m;

        row_sum_num+=m;

        ratio = 1.0l*total_csr/row_sum_num;

        if (ratio > 0.05 || 1.0l*col_num[i].num/m > 0.5 ||total_csr * 4 > total_dense){
            break;
        }

        densecolnum--;
        densefg[col_num[i].x]= false;
    }

    //initialization dense and sparse

    if(densecolnum>0){
        dense.val.resize(m*densecolnum);
        dense.row=m;
        dense.col=densecolnum;
        dense.Ttable=new int[n];
        dense.is_use = true;

        kernel_type *h_dense = dense.val.host_data();
        
        for (int i=0,p_row=0;i<n;i++){
            if (densefg[i]==true){

                dense.Ttable[i]=p_row;
                p_row++;
               
            }
        }

        #pragma omp parallel for 
        for (int i=0;i<m;i++){
            int csr_row_begin = csr_row_ptr[i];
            int csr_row_end = csr_row_ptr[i+1];

            for (int j=csr_row_begin;j<csr_row_end;j++) {
                if (densefg[csr_col_ind[j]] == true) {

                    h_dense[i * densecolnum + dense.Ttable[csr_col_ind[j]]] = csr_val[j];
                    
                }
            }
        }
    }
    
    //sparse
    std::vector<kernel_type> val_data;
    std::vector<int> row_ptr;
    std::vector<int> col_ptr;

    val_data.clear();
    col_ptr.clear();
    row_ptr.clear();
    row_ptr.push_back(0);

    sparse.row=m;
    sparse.col=n;
    //csr
    if(densecolnum<n){
        sparse.is_use = true;
        for (int i=0;i<m;i++){
            int csr_row_begin = csr_row_ptr[i];
            int csr_row_end = csr_row_ptr[i+1];
            for (int j=csr_row_begin;j<csr_row_end;j++){
                if (densefg[csr_col_ind[j]]==false){

                    val_data.push_back(csr_val[j]);
                    col_ptr.push_back(csr_col_ind[j]);
                    
                }
            }

            row_ptr.push_back(val_data.size());   

        }
        sparse.val_.resize(val_data.size());
        sparse.col_ind_.resize(col_ptr.size());
        sparse.row_ptr_.resize(row_ptr.size());

        sparse.val_.copy_from(val_data.data(), val_data.size());
        sparse.col_ind_.copy_from(col_ptr.data(), col_ptr.size());
        sparse.row_ptr_.copy_from(row_ptr.data(), row_ptr.size());
    }
    



    delete[] col_num;
    delete[] densefg;
    

}





void
CSMOSolver::solve(const KernelMatrix &k_mat, const SyncArray<int> &y, SyncArray<float_type> &alpha, float_type &rho,
                  SyncArray<float_type> &f_val, float_type eps, float_type Cp, float_type Cn, int ws_size,
                  int out_max_iter) const {
    int n_instances = k_mat.n_instances();
    int q = ws_size / 2;


    SyncArray<int> working_set(ws_size);
    SyncArray<int> working_set_first_half(q);
    SyncArray<int> working_set_last_half(q);
#ifdef USE_CUDA
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
#endif
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncArray<int> f_idx(n_instances);
    SyncArray<int> f_idx2sort(n_instances);
    SyncArray<float_type> f_val2sort(n_instances);
    SyncArray<float_type> alpha_diff(ws_size);
    SyncArray<float_type> diff(2);

    SyncArray<kernel_type> k_mat_rows(ws_size * k_mat.n_instances());
    SyncArray<kernel_type> k_mat_rows_first_half(q * k_mat.n_instances());
    SyncArray<kernel_type> k_mat_rows_last_half(q * k_mat.n_instances());
#ifdef USE_CUDA
    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * k_mat.n_instances()]);
#else
    k_mat_rows_first_half.set_host_data(k_mat_rows.host_data());
    k_mat_rows_last_half.set_host_data(&k_mat_rows.host_data()[q * k_mat.n_instances()]);
#endif
    int *f_idx_data = f_idx.host_data();
    for (int i = 0; i < n_instances; ++i) {
        f_idx_data[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG(INFO) << "training start";
    int max_iter = max(100000, ws_size > INT_MAX / 100 ? INT_MAX : 100 * ws_size);
    long long local_iter = 0;

    //avoid infinite loop of repeated local diff
    int same_local_diff_cnt = 0;
    float_type previous_local_diff = INFINITY;
    int swap_local_diff_cnt = 0;
    float_type last_local_diff = INFINITY;
    float_type second_last_local_diff = INFINITY;
   

    //矩阵划分
    long long part = 0;
    TDEF(part)
    TSTART(part)
    SparseData sparse_mat;
    DenseData dense_mat;
    CSR_DenseCSR(k_mat,dense_mat,sparse_mat);
    TEND(part)
    part+=TINT(part);
    LOG(INFO)<<"sparse matrix shape is "<<sparse_mat.row<<" "<<sparse_mat.col<<" "<<sparse_mat.is_use;
    LOG(INFO)<<"dense matrix shape is "<<dense_mat.row<<" "<<dense_mat.col<<" "<<dense_mat.is_use;
    LOG(INFO)<<"matrix partition time is  "<<part/1e6;


    long long select_time = 0;
    long long local_smo_time = 0;
    long long get_rows_time = 0;
    TDEF(CSMOSolver)
    TDEF(select)
    TDEF(getrows)
    TDEF(local_smo)

    TSTART(CSMOSolver)
    for (int iter = 0;; ++iter) {
        //select working set
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
        sort_f(f_val2sort, f_idx2sort);
        vector<int> ws_indicator(n_instances, 0);
        if (0 == iter) {
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set);
            TSTART(getrows)
            //k_mat.get_rows(working_set, k_mat_rows);

            //test new funciton 
            //SyncArray<kernel_type> k_test(ws_size * k_mat.n_instances());
            k_mat.get_sparse_dense_rows(working_set,sparse_mat,dense_mat,k_mat_rows);

            //check
            //kernel_type* t1 = k_mat_rows.host_data();
            //kernel_type* t2 = k_test.host_data();
            //float r = 0.0;
            //for (int i = 0; i < 10; ++i)
            //{
                /* code */
              //  r+=fabs(t1[i]-t2[i]);
            //}
            //LOG(INFO)<<"diff is "<<r;
            TEND(getrows)
            get_rows_time +=TINT(getrows);
        } else {
            working_set_first_half.copy_from(working_set_last_half);
            int *working_set_data = working_set.host_data();
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set_data[i]] = 1;
            }
            select_working_set(ws_indicator, f_idx2sort, y, alpha, Cp, Cn, working_set_last_half);
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
            TSTART(getrows)
            k_mat.get_sparse_dense_rows(working_set_last_half,sparse_mat,dense_mat,k_mat_rows_last_half);
            //k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
            TEND(getrows)
            get_rows_time += TINT(getrows);
        }
        //local smo
        TSTART(local_smo)
        smo_kernel(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat.diag(), n_instances, eps, diff,
                   max_iter);
        TEND(local_smo)
        local_smo_time+=TINT(local_smo);
        //update f
        update_f(f_val, alpha_diff, k_mat_rows, k_mat.n_instances());
        float_type *diff_data = diff.host_data();
        local_iter += diff_data[1];

        //track unchanged diff
        if (fabs(diff_data[0] - previous_local_diff) < eps * 0.001) {
            same_local_diff_cnt++;
        } else {
            same_local_diff_cnt = 0;
            previous_local_diff = diff_data[0];
        }

        //track unchanged swapping diff
        if(fabs(diff_data[0] - second_last_local_diff) < eps * 0.001){
            swap_local_diff_cnt++;
        } else {
            swap_local_diff_cnt = 0;
        }
        second_last_local_diff = last_local_diff;
        last_local_diff = diff_data[0];

        if (iter % 100 == 0)
            LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                      << diff_data[0];
        //todo find some other ways to deal unchanged diff
        //training terminates in three conditions: 1. diff stays unchanged; 2. diff is closed to 0; 3. training reaches the limit of iterations.
        //repeatedly swapping between two diffs
        if ((same_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps) || diff_data[0] < eps ||
            (out_max_iter != -1) && (iter == out_max_iter) ||
            (swap_local_diff_cnt >= 10 && fabs(diff_data[0] - 2.0) > eps)) {
            rho = calculate_rho(f_val, y, alpha, Cp, Cn);
            LOG(INFO) << "global iter = " << iter << ", total local iter = " << local_iter << ", diff = "
                      << diff_data[0];
            LOG(INFO) << "training finished";
            float_type obj = calculate_obj(f_val, alpha, y);
            LOG(INFO) << "obj = " << obj;
            break;
        }
    }
    
    LOG(INFO)<<"get rows time is "<<get_rows_time/1e6;
    LOG(INFO)<<"local smo time is "<<local_smo_time/1e6;
    LOG(INFO)<<"trans time is "<<time1/1e6;
    TEND(CSMOSolver)
    TPRINT(CSMOSolver,"loop time is :")


}

void
CSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncArray<int> &f_idx2sort, const SyncArray<int> &y,
                               const SyncArray<float_type> &alpha, float_type Cp, float_type Cn,
                               SyncArray<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    int *working_set_data = working_set.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !is_I_up(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_up
                p_left++;
                if (p_left == n_instances) break;
                i = index[p_left];
            }
            if (p_left < n_instances) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !is_I_low(alpha_data[i], y_data[i], Cp, Cn)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set_data[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }

    }
}

float_type
CSMOSolver::calculate_rho(const SyncArray<float_type> &f_val, const SyncArray<int> &y, SyncArray<float_type> &alpha,
                          float_type Cp,
                          float_type Cn) const {
    int n_free = 0;
    double sum_free = 0;
    float_type up_value = INFINITY;
    float_type low_value = -INFINITY;
    const float_type *f_val_data = f_val.host_data();
    const int *y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (is_free(alpha_data[i], y_data[i], Cp, Cn)) {
            n_free++;
            sum_free += f_val_data[i];
        }
        if (is_I_up(alpha_data[i], y_data[i], Cp, Cn)) up_value = min(up_value, f_val_data[i]);
        if (is_I_low(alpha_data[i], y_data[i], Cp, Cn)) low_value = max(low_value, f_val_data[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void CSMOSolver::init_f(const SyncArray<float_type> &alpha, const SyncArray<int> &y, const KernelMatrix &k_mat,
                        SyncArray<float_type> &f_val) const {
    //todo auto set batch size
    int batch_size = 100;
    vector<int> idx_vec;
    vector<float_type> alpha_diff_vec;
    const int *y_data = y.host_data();
    const float_type *alpha_data = alpha.host_data();
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha_data[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha_data[i] * y_data[i]);
        }
        if (idx_vec.size() > batch_size || (i == alpha.size() - 1 && !idx_vec.empty())) {
            SyncArray<int> idx(idx_vec.size());
            SyncArray<float_type> alpha_diff(idx_vec.size());
            idx.copy_from(idx_vec.data(), idx_vec.size());
            alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
            SyncArray<kernel_type> kernel_rows(idx.size() * k_mat.n_instances());
            k_mat.get_rows(idx, kernel_rows);
            update_f(f_val, alpha_diff, kernel_rows, k_mat.n_instances());
            idx_vec.clear();
            alpha_diff_vec.clear();
        }
    }
}

void
CSMOSolver::smo_kernel(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                       SyncArray<float_type> &alpha_diff,
                       const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                       const SyncArray<kernel_type> &k_mat_rows,
                       const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps,
                       SyncArray<float_type> &diff,
                       int max_iter) const {
    c_smo_solve(y, f_val, alpha, alpha_diff, working_set, Cp, Cn, k_mat_rows, k_mat_diag, row_len, eps, diff, max_iter);
}

float_type CSMOSolver::calculate_obj(const SyncArray<float_type> &f_val, const SyncArray<float_type> &alpha,
                                     const SyncArray<int> &y) const {
    //todo use parallel reduction for gpu and cpu
    int n_instances = f_val.size();
    float_type obj = 0;
    const float_type *f_val_data = f_val.host_data();
    const float_type *alpha_data = alpha.host_data();
    const int *y_data = y.host_data();
    for (int i = 0; i < n_instances; ++i) {
        obj += alpha_data[i] - (f_val_data[i] + y_data[i]) * alpha_data[i] * y_data[i] / 2;
    }
    return -obj;
}

