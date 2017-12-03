
//
// Created by jiashuai on 17-11-7.
//


#include <thundersvm/kernel/smo_kernel.h>
#include <omp.h>
#ifndef USE_CUDA
namespace svm_kernel {
    int get_min_idx(const float *values, int size) {
        int min_idx = 0;
        float min = INFINITY;
        for (int i = 0; i < size; ++i) {
            if (values[i] < min) {
                min = values[i];
                min_idx = i;
            }
        }
        return min_idx;
    }
  
    int get_block_min(const float *values, int *index){
    int tid = omp_get_thread_num();
    index[tid] = tid;
#pragma omp barrier
    for(int offset = omp_get_num_threads() / 2; offset > 0; offset >>= 1){
        if(tid < offset) {
            if(values[index[tid + offset]] <= values[index[tid]]) {
                index[tid] = index[tid + offset];
            }
        }
#pragma omp barrier
    }
    return index[0];
    }


    void c_smo_solve(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                     SyncData<float_type> &alpha_diff,
                     const SyncData<int> &working_set, float_type Cp, float_type Cn,
                     const SyncData<float_type> &k_mat_rows,
                     const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                     int max_iter) {
        //todo rewrite these codes
        //allocate shared memory
        //const int *y_ptr = y.host_data();
        //float_type* f_val_ptr = f_val.host_data();
        float_type* alpha_ptr = alpha.host_data();
        float_type* alpha_diff_ptr = alpha_diff.host_data();


        int ws_size = working_set.size();
        int *shared_mem = new int[ws_size * 3 * sizeof(float) + 2 * sizeof(float)];
        //int *shared_mem = (int *)malloc(ws_size * (2 * sizeof(float) + sizeof(int)) + 2 * sizeof(float));
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float *a_old = new float[ws_size];
        int *wsi = new int[ws_size];
        float *a_ = new float[ws_size];
        float *kIwsI = new float[ws_size];
        //float *kJ2wsI = new float[ws_size];
        float *f_ = new float[ws_size];
        float *y_ = new float[ws_size];
        /*
        for(int idx = 0; idx < ws_size; idx++){
            wsi[idx] = working_set[idx];
            kd[idx] = k_mat_diag[wsi[idx]];
            y_[idx] = y[wsi[idx]];
            f_[idx] = f_val[wsi[idx]];
            a_[idx] = alpha[wsi[idx]];
            a_old[idx] = a_[idx];
        }
         */
        const int nthread = std::min(ws_size, omp_get_max_threads());
        //const int nthread = 2;
        //float local_eps;
        //bool go = true;
        //float local_diff;
        //int numOfIter = 0;
        /*
        int i, j1, j2;
        float l;
        float up_value, low_value;
        float local_eps, local_diff;
        int numOfIter = 0;
         */
    #pragma omp parallel num_threads(nthread)
    {
        //bool go = true;
        //float local_eps;
        int i, j1, j2;
        float l;
        float up_value, low_value;
        float local_eps, local_diff;
        int numOfIter = 0;
        int tid = omp_get_thread_num();
        int step = (ws_size + nthread - 1) / nthread;
        int begin = std::min(tid * step, ws_size);
        int end = std::min((tid + 1) * step, ws_size);
        //std::cout<<"begin"<<begin<<std::endl;
        //std::cout<<"end"<<end<<std::endl;

        for(int idx = begin; idx < end; idx++){
            wsi[idx] = working_set[idx];
            kd[idx] = k_mat_diag[wsi[idx]];
            y_[idx] = y[wsi[idx]];
            f_[idx] = f_val[wsi[idx]];
            a_[idx] = alpha[wsi[idx]];
            a_old[idx] = a_[idx];
        }

    //#pragma omp barrier
        //int numOfIter = 0;
        while (1) {
#pragma omp barrier
            //select fUp and fLow
            for (int idx = begin; idx < end; ++idx) {
                if (is_I_up(a_[idx], y_[idx], Cp, Cn))
                    f_val2reduce[idx] = f_[idx];
                else
                    f_val2reduce[idx] = INFINITY;
            }
#pragma omp barrier
            //if (tid == 0) {
                i = get_min_idx(f_val2reduce, ws_size);
                //int i = get_block_min(f_val2reduce, f_idx2reduce);
                up_value = f_val2reduce[i];
            //}
//#pragma omp barrier
            for (int idx = begin; idx < end; ++idx) {
                kIwsI[idx] = k_mat_rows[row_len * i + wsi[idx]];//K[i, wsi]
            }
#pragma omp barrier
            for (int idx = begin; idx < end; ++idx) {
                if (is_I_low(a_[idx], y_[idx], Cp, Cn))
                    f_val2reduce[idx] = -f_[idx];
                else
                    f_val2reduce[idx] = INFINITY;
            }
#pragma omp barrier
                j1 = get_min_idx(f_val2reduce, ws_size);
            //int j1 = get_block_min(f_val2reduce, f_idx2reduce);
                low_value = -f_val2reduce[j1];
                local_diff = low_value - up_value;
            if (numOfIter == 0) {

                //if(tid == 0){
                    local_eps = max(eps, 0.1f * local_diff);
                //if(tid == 0)
#pragma omp single
                {
                    diff[0] = local_diff;
                }
                    //printf("tid0:%f\n", local_eps);
                //if(diff[0] != 0)
                    //printf("diff[0]:%f\n", diff[0]);
                //}
            }
            //}
//#pragma omp barrier
            //if(tid == 1)
            //    printf("tid1:%f\n", local_eps);
            if (local_diff < local_eps) {
                for (int idx = begin; idx < end; idx++) {
                    //int wsi = working_set[idx];
                    //float a_diff = a_old[idx] - a_[idx];
                    alpha_ptr[wsi[idx]] = a_[idx];
                    alpha_diff_ptr[idx] = (a_old[idx] - a_[idx]) * y_[idx];
                    //alpha_diff[idx] = -(alpha[wsi[idx]] - a_old[idx]) * y_[idx];
                    /*
                    if(idx == 0 || idx == 32 || idx == 77)
                        printf("a_[%d]:%f a_old[%d]:%f y_[%d]:%f alpha_diff[%d]:%f\n", idx, a_[idx], idx, a_old[idx], idx, y_[idx], idx, alpha_diff[idx]);
                    if(alpha[wsi[idx]] != 0)
                        printf("alpha[%d]:%f\n", wsi[idx], alpha[wsi[idx]]);
                    if(alpha_diff[idx] != 0 || idx == 32 || idx == 77){
                        printf("alpha_diff[%d]:%f\n", idx, alpha_diff[idx]);
                    }
                    */
                }
                /*
                if(begin != end){
                    alpha[wsi[begin]] = a_[begin];
                    alpha_diff[begin] = (a_old[begin] - a_[begin]) * y_[begin];
                }
                 */
                break;
            }
#pragma omp barrier
            //if(go){
                //printf("after first break\n");
            for (int idx = begin; idx < end; ++idx) {
                //int wsi = working_set[idx];
                if (-up_value > -f_[idx] && (is_I_low(a_[idx], y_[idx], Cp, Cn))) {
                    float aIJ = kd[i] + kd[idx] - 2 * kIwsI[idx];
                    float bIJ = -up_value + f_[idx];
                    f_val2reduce[idx] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[idx] = INFINITY;
            }
#pragma omp barrier
            //if(tid == 0) {
                j2 = get_min_idx(f_val2reduce, ws_size);
            if(tid == 0) {
//#pragma omp single
//{
                *alpha_i_diff = y_[i] > 0 ? Cp - a_[i] : a_[i];
                *alpha_j_diff = min(y_[j2] > 0 ? a_[j2] : Cn - a_[j2],
                                    (-up_value + f_[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
}
            //}
#pragma omp barrier
                l = min(*alpha_i_diff, *alpha_j_diff);
            if(tid == 0) {
//#pragma omp single
  //          {
                a_[i] += l * y_[i];
                a_[j2] -= l * y_[j2];
            }

//#pragma omp barrier
            //update f
            for (int idx = begin; idx < end; ++idx) {
                float kJ2wsI = k_mat_rows[row_len * j2 + wsi[idx]];//K[J2, wsi]
                f_[idx] -= l * (kJ2wsI - kIwsI[idx]);
            }
            //if(tid == 0)
            numOfIter++;

            if (numOfIter > max_iter)
                //go = false;
                 break;
            //}
        }
//#pragma omp barrier
    }
    /*
        delete[] a_old;
        delete[] a_;
        delete[] wsi;
        delete[] f_;
        delete[] kIwsI;
        //delete[] kJ2wsI;
        delete[] y_;
        delete[] shared_mem;
        //printf("end of c_smo_solve\n");
        */
    }

    void nu_smo_solve(const SyncData<int> &y, SyncData<float_type> &f_val, SyncData<float_type> &alpha,
                      SyncData<float_type> &alpha_diff,
                      const SyncData<int> &working_set, float_type C, const SyncData<float_type> &k_mat_rows,
                      const SyncData<float_type> &k_mat_diag, int row_len, float_type eps, SyncData<float_type> &diff,
                      int max_iter) {
        //allocate shared memory
        int ws_size = working_set.size();
        int *shared_mem = new int[ws_size * 3 + 2];
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float *a_old = new float[ws_size];
        float *kIpwsI = new float[ws_size];
        float *kInwsI = new float[ws_size];
        float *f = new float[ws_size];
        for (int tid = 0; tid < ws_size; ++tid) {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = alpha[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float local_eps;
        int numOfIter = 0;
        while (1) {
            //select I_up (y=+1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] < C)
                    f_val2reduce[tid] = f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int ip = get_min_idx(f_val2reduce, ws_size);
            float up_value_p = f_val2reduce[ip];
            for (int tid = 0; tid < ws_size; ++tid) {
                kIpwsI[tid] = k_mat_rows[row_len * ip + working_set[tid]];//K[i, wsi]
            }

            //select I_up (y=-1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] > 0)
                    f_val2reduce[tid] = f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int in = get_min_idx(f_val2reduce, ws_size);
            float up_value_n = f_val2reduce[in];
            for (int tid = 0; tid < ws_size; ++tid) {
                kInwsI[tid] = k_mat_rows[row_len * in + working_set[tid]];//K[i, wsi]
            }

            //select I_low (y=+1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] > 0)
                    f_val2reduce[tid] = -f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int j1p = get_min_idx(f_val2reduce, ws_size);
            float low_value_p = -f_val2reduce[j1p];

            //select I_low (y=-1)
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] < C)
                    f_val2reduce[tid] = -f[tid];
                else
                    f_val2reduce[tid] = INFINITY;
            }
            int j1n = get_min_idx(f_val2reduce, ws_size);
            float low_value_n = -f_val2reduce[j1n];

            float local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);

            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                diff[0] = local_diff;
            }
            
            if (local_diff < local_eps) {
                //printf("before alpha_diff\n");
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];

                }
                //printf("after alpha_diff\n");
                break;
            }

            //select j2p using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_p > -f[tid] && y[wsi] > 0 && alpha[wsi] > 0) {
                    float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float bIJ = -up_value_p + f[tid];
                    f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[tid] = INFINITY;
            }
            int j2p = get_min_idx(f_val2reduce, ws_size);
            float f_val_j2p = f_val2reduce[j2p];

            //select j2n using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_n > -f[tid] && y[wsi] < 0 && alpha[wsi] < C) {
                    float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float bIJ = -up_value_n + f[tid];
                    f_val2reduce[tid] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[tid] = INFINITY;
            }
            int j2n = get_min_idx(f_val2reduce, ws_size);

            int i, j2;
            float up_value;
            float *kIwsI;
            if (f_val_j2p < f_val2reduce[j2n]) {
                i = ip;
                j2 = j2p;
                up_value = up_value_p;
                kIwsI = kIpwsI;
            } else {
                i = in;
                j2 = j2n;
                kIwsI = kInwsI;
                up_value = up_value_n;
            }
            //update alpha
//            if (tid == i)
//                *alpha_i_diff = y > 0 ? C - a : a;
            *alpha_i_diff = y[working_set[i]] > 0 ? C - alpha[working_set[i]] : alpha[working_set[i]];
//            if (tid == j2)
//                *alpha_j_diff = min(y > 0 ? a : C - a, (-up_value + f) / (kd[i] + kd[j2] - 2 * kIwsI));
            *alpha_j_diff = min(y[working_set[j2]] > 0 ? alpha[working_set[j2]] : C - alpha[working_set[j2]],
                                (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
//            __syncthreads();
            float l = min(*alpha_i_diff, *alpha_j_diff);

            alpha[working_set[i]] += l * y[working_set[i]];
            alpha[working_set[j2]] -= l * y[working_set[j2]];

            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                float kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
            if (numOfIter > max_iter) break;
        }
        delete[] a_old;
        delete[] f;
        delete[] kIpwsI;
        delete[] kInwsI;
        delete[] shared_mem;
        //printf("after nu_smo_solve\n");
    }

    void
    update_f(SyncData<float_type> &f, const SyncData<float_type> &alpha_diff, const SyncData<float_type> &k_mat_rows,
             int n_instances) {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; ++idx) {
            float_type sum_diff = 0;
            for (int i = 0; i < alpha_diff.size(); ++i) {
                float_type d = alpha_diff[i];
                if (d != 0) {
                    sum_diff += d * k_mat_rows[i * n_instances + idx];
                }
            }
            f[idx] -= sum_diff;
        }
    }

    void sort_f(SyncData<float_type> &f_val2sort, SyncData<int> &f_idx2sort) {
        vector<std::pair<float_type, int>> paris;
        for (int i = 0; i < f_val2sort.size(); ++i) {
            paris.emplace_back(f_val2sort[i], f_idx2sort[i]);
        }
        std::sort(paris.begin(), paris.end());
        for (int i = 0; i < f_idx2sort.size(); ++i) {
            f_idx2sort[i] = paris[i].second;
        }
    }
}
#endif

