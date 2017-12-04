
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


    void c_smo_solve_kernel(const int *y, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                            const int *working_set,
                            int ws_size,
                            float Cp, float Cn, const float *k_mat_rows, const float *k_mat_diag, int row_len,
                            float_type eps,
                            float_type *diff, int max_iter) {
        //allocate shared memory

        int *shared_mem = new int[ws_size * 3 * sizeof(float) + 2 * sizeof(float)];
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
        float *f_ = new float[ws_size];
        float *y_ = new float[ws_size];
        const int nthread = std::min(ws_size, omp_get_max_threads());
        //const int nthread = 2;
    #pragma omp parallel num_threads(nthread)
    {
        int i, j1, j2;
        float l;
        float up_value, low_value;
        float local_eps, local_diff;
        int numOfIter = 0;
        int tid = omp_get_thread_num();
        int step = (ws_size + nthread - 1) / nthread;
        int begin = std::min(tid * step, ws_size);
        int end = std::min((tid + 1) * step, ws_size);

        for(int idx = begin; idx < end; idx++){
            wsi[idx] = working_set[idx];
            kd[idx] = k_mat_diag[wsi[idx]];
            y_[idx] = y[wsi[idx]];
            f_[idx] = f_val[wsi[idx]];
            a_[idx] = alpha[wsi[idx]];
            a_old[idx] = a_[idx];
        }

        while (1) {
            //select fUp and fLow
            for (int idx = begin; idx < end; ++idx) {
                if (is_I_up(a_[idx], y_[idx], Cp, Cn))
                    f_val2reduce[idx] = f_[idx];
                else
                    f_val2reduce[idx] = INFINITY;
            }
#pragma omp barrier
            i = get_min_idx(f_val2reduce, ws_size);
            up_value = f_val2reduce[i];
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
                low_value = -f_val2reduce[j1];
                local_diff = low_value - up_value;
            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                if(tid == 0)
                    diff[0] = local_diff;
            }
            if (local_diff < local_eps) {
                for (int idx = begin; idx < end; idx++) {
                    alpha[wsi[idx]] = a_[idx];
                    alpha_diff[idx] = (a_old[idx] - a_[idx]) * y_[idx];
                }
                break;
            }
#pragma omp barrier
            for (int idx = begin; idx < end; ++idx) {
                if (-up_value > -f_[idx] && (is_I_low(a_[idx], y_[idx], Cp, Cn))) {
                    float aIJ = kd[i] + kd[idx] - 2 * kIwsI[idx];
                    float bIJ = -up_value + f_[idx];
                    f_val2reduce[idx] = -bIJ * bIJ / aIJ;
                } else
                    f_val2reduce[idx] = INFINITY;
            }
#pragma omp barrier
                j2 = get_min_idx(f_val2reduce, ws_size);
            if(tid == 0) {
                *alpha_i_diff = y_[i] > 0 ? Cp - a_[i] : a_[i];
                *alpha_j_diff = min(y_[j2] > 0 ? a_[j2] : Cn - a_[j2],
                                    (-up_value + f_[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            }
#pragma omp barrier
                l = min(*alpha_i_diff, *alpha_j_diff);
            if(tid == 0) {
                a_[i] += l * y_[i];
                a_[j2] -= l * y_[j2];
            }
            //update f
            for (int idx = begin; idx < end; ++idx) {
                float kJ2wsI = k_mat_rows[row_len * j2 + wsi[idx]];//K[J2, wsi]
                f_[idx] -= l * (kJ2wsI - kIwsI[idx]);
            }
            numOfIter++;

            if (numOfIter > max_iter)
                 break;
        }
    }

        delete[] a_old;
        delete[] a_;
        delete[] wsi;
        delete[] f_;
        delete[] kIwsI;
        delete[] y_;
        delete[] shared_mem;

    }

    void c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                     SyncArray<float_type> &alpha_diff,
                     const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                     const SyncArray<float_type> &k_mat_rows,
                     const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                     int max_iter) {
        c_smo_solve_kernel(y.host_data(), f_val.host_data(), alpha.host_data(), alpha_diff.host_data(),
                           working_set.host_data(), working_set.size(), Cp, Cn, k_mat_rows.host_data(),
                           k_mat_diag.host_data(), row_len, eps, diff.host_data(), max_iter);
    }

    void nu_smo_solve_kernel(const int *y, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                        const int *working_set,
                        int ws_size, float C, const float *k_mat_rows, const float *k_mat_diag, int row_len,
                        float_type eps,
                        float_type *diff, int max_iter) {
        //allocate shared memory
        int *shared_mem = new int[ws_size * 3 + 2];
        int *f_idx2reduce = shared_mem; //temporary memory for reduction
        float *f_val2reduce = (float *) &f_idx2reduce[ws_size]; //f values used for reduction.
        float *alpha_i_diff = &f_val2reduce[ws_size]; //delta alpha_i
        float *alpha_j_diff = &alpha_i_diff[1];
        float *kd = &alpha_j_diff[1]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float *a_old = new float[ws_size];
        int *wsi = new int[ws_size];
        float *a_ = new float[ws_size];
        float *kIpwsI = new float[ws_size];
        float *kInwsI = new float[ws_size];
        float *kIwsI = new float[ws_size];
        float *f_ = new float[ws_size];
        float *y_ = new float[ws_size];
        const int nthread = std::min(ws_size, omp_get_max_threads());
        //const int nthread = 2;
#pragma omp parallel num_threads(nthread)
        {
            //bool go = true;
            //float local_eps;
            int ip, in, j1p, j1n, j2p, j2n;
            float f_val_j2p;
            float l;
            float up_value_p, up_value_n, low_value_p, low_value_n;
            float local_eps, local_diff;
            int numOfIter = 0;
            int tid = omp_get_thread_num();
            int step = (ws_size + nthread - 1) / nthread;
            int begin = std::min(tid * step, ws_size);
            int end = std::min((tid + 1) * step, ws_size);

            for(int idx = begin; idx < end; idx++){
                wsi[idx] = working_set[idx];
                kd[idx] = k_mat_diag[wsi[idx]];
                y_[idx] = y[wsi[idx]];
                f_[idx] = f_val[wsi[idx]];
                a_[idx] = alpha[wsi[idx]];
                a_old[idx] = a_[idx];
            }
            while (1) {
                for (int idx = begin; idx < end; ++idx) {
                    if (y_[idx] > 0 && a_[idx] < C)
                        f_val2reduce[idx] = f_[idx];
                    else
                        f_val2reduce[idx] = INFINITY;
                }
#pragma omp barrier
                ip = get_min_idx(f_val2reduce, ws_size);
                up_value_p = f_val2reduce[ip];
                for (int idx = begin; idx < end; ++idx) {
                    kIpwsI[idx] = k_mat_rows[row_len * ip + wsi[idx]];//K[i, wsi]
                }
#pragma omp barrier
                for (int idx = begin; idx < end; ++idx) {
                    if (y_[idx] < 0 && a_[idx] > 0)
                        f_val2reduce[idx] = f_[idx];
                    else
                        f_val2reduce[idx] = INFINITY;
                }
#pragma omp barrier
                in = get_min_idx(f_val2reduce, ws_size);
                up_value_n = f_val2reduce[in];
                for (int idx = begin; idx < end; ++idx) {
                    kInwsI[idx] = k_mat_rows[row_len * in + wsi[idx]];//K[i, wsi]
                }
#pragma omp barrier
                for (int idx = begin; idx < end; ++idx) {
                    if (y_[idx] > 0 && a_[idx] > 0)
                        f_val2reduce[idx] = -f_[idx];
                    else
                        f_val2reduce[idx] = INFINITY;
                }
#pragma omp barrier
                j1p = get_min_idx(f_val2reduce, ws_size);
                low_value_p = -f_val2reduce[j1p];
#pragma omp barrier
                for (int idx = begin; idx < end; ++idx) {
                    if (y_[idx] < 0 && a_[idx] < C)
                        f_val2reduce[idx] = -f_[idx];
                    else
                        f_val2reduce[idx] = INFINITY;
                }
#pragma omp barrier
                j1n = get_min_idx(f_val2reduce, ws_size);
                low_value_n = -f_val2reduce[j1n];


                local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);
                if (numOfIter == 0) {
                    local_eps = max(eps, 0.1f * local_diff);
                    if(tid == 0)
                    {
                        diff[0] = local_diff;
                    }
                }
                if (local_diff < local_eps) {
                    for (int idx = begin; idx < end; idx++) {
                        alpha[wsi[idx]] = a_[idx];
                        alpha_diff[idx] = (a_old[idx] - a_[idx]) * y_[idx];
                    }
                    break;
                }
#pragma omp barrier
                for (int idx = begin; idx < end; ++idx) {
                    if (-up_value_p > -f_[idx] && y_[idx] > 0 && a_[idx] > 0) {
                        float aIJ = kd[ip] + kd[idx] - 2 * kIpwsI[idx];
                        float bIJ = -up_value_p + f_[idx];
                        f_val2reduce[idx] = -bIJ * bIJ / aIJ;
                    } else
                        f_val2reduce[idx] = INFINITY;
                }
#pragma omp barrier
                j2p = get_min_idx(f_val2reduce, ws_size);
                f_val_j2p = f_val2reduce[j2p];
#pragma omp barrier
                for (int idx = begin; idx < end; ++idx) {
                    if (-up_value_n > -f_[idx] && y_[idx] < 0 && a_[idx] < C) {
                        float aIJ = kd[in] + kd[idx] - 2 * kInwsI[idx];
                        float bIJ = -up_value_n + f_[idx];
                        f_val2reduce[idx] = -bIJ * bIJ / aIJ;
                    } else
                        f_val2reduce[idx] = INFINITY;
                }
#pragma omp barrier
                j2n = get_min_idx(f_val2reduce, ws_size);
                int i, j2;
                float up_value;

                if (f_val_j2p < f_val2reduce[j2n]) {
                    i = ip;
                    j2 = j2p;
                    up_value = up_value_p;
                    for(int idx = begin; idx < end; ++idx)
                        kIwsI[idx] = kIpwsI[idx];
                } else {
                    i = in;
                    j2 = j2n;
                    up_value = up_value_n;
                    for(int idx = begin; idx < end; ++idx)
                        kIwsI[idx] = kInwsI[idx];
                }




                if(tid == 0) {
                    *alpha_i_diff = y_[i] > 0 ? C - a_[i] : a_[i];
                    *alpha_j_diff = min(y_[j2] > 0 ? a_[j2] : C - a_[j2],
                                        (-up_value + f_[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
                }
#pragma omp barrier
                l = min(*alpha_i_diff, *alpha_j_diff);
                if(tid == 0) {
                    a_[i] += l * y_[i];
                    a_[j2] -= l * y_[j2];
                }
                //update f
                for (int idx = begin; idx < end; ++idx) {
                    float kJ2wsI = k_mat_rows[row_len * j2 + wsi[idx]];//K[J2, wsi]
                    f_[idx] -= l * (kJ2wsI - kIwsI[idx]);
                }
                numOfIter++;
                if (numOfIter > max_iter)
                    break;
            }
        }

        delete[] a_old;
        delete[] a_;
        delete[] wsi;
        delete[] f_;
        delete[] kIwsI;
        delete[] kIpwsI;
        delete[] kInwsI;
        delete[] y_;
        delete[] shared_mem;
    }
    void nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                      SyncArray<float_type> &alpha_diff,
                      const SyncArray<int> &working_set, float_type C, const SyncArray<float_type> &k_mat_rows,
                      const SyncArray<float_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                      int max_iter) {
        nu_smo_solve_kernel(y.host_data(), f_val.host_data(), alpha.host_data(), alpha_diff.host_data(),
                           working_set.host_data(), working_set.size(), C, k_mat_rows.host_data(),
                           k_mat_diag.host_data(), row_len, eps, diff.host_data(), max_iter);
    }

    void
    update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff, const SyncArray<float_type> &k_mat_rows,
             int n_instances) {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        float_type *f_data = f.host_data();
        const float_type *alpha_diff_data = alpha_diff.host_data();
        const float_type *k_mat_rows_data = k_mat_rows.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; ++idx) {
            float_type sum_diff = 0;
            for (int i = 0; i < alpha_diff.size(); ++i) {
                float_type d = alpha_diff_data[i];
                if (d != 0) {
                    sum_diff += d * k_mat_rows_data[i * n_instances + idx];
                }
            }
            f_data[idx] -= sum_diff;
        }
    }

    void sort_f(SyncArray<float_type> &f_val2sort, SyncArray<int> &f_idx2sort) {
        vector<std::pair<float_type, int>> paris;
        float_type *f_val2sort_data = f_val2sort.host_data();
        int *f_idx2sort_data = f_idx2sort.host_data();
        for (int i = 0; i < f_val2sort.size(); ++i) {
            paris.emplace_back(f_val2sort_data[i], f_idx2sort_data[i]);
        }
        std::sort(paris.begin(), paris.end());
        for (int i = 0; i < f_idx2sort.size(); ++i) {
            f_idx2sort_data[i] = paris[i].second;
        }
    }
}
#endif

