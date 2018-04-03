
//
// Created by jiashuai on 17-11-7.
//


#include <thundersvm/kernel/smo_kernel.h>
#include <omp.h>

namespace svm_kernel {

    void c_smo_solve_kernel(const int *label, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                            const int *working_set,
                            int ws_size,
                            float_type Cp, float_type Cn, const kernel_type *k_mat_rows, const kernel_type *k_mat_diag,
                            int row_len,
                            float_type eps,
                            float_type *diff, int max_iter) {
        //allocate shared memory
        float_type alpha_i_diff; //delta alpha_i
        float_type alpha_j_diff;
        vector<kernel_type> kd(ws_size); // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        vector<float_type> a_old(ws_size);
        vector<float_type> kIwsI(ws_size);
        vector<float_type> f(ws_size);
        vector<float_type> y(ws_size);
        vector<float_type> a(ws_size);
        for (int tid = 0; tid < ws_size; ++tid) {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = a[tid] = alpha[wsi];
            y[tid] = label[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float_type local_eps;
        int numOfIter = 0;
        while (1) {
            //select fUp and fLow
            int i = 0;
            float_type up_value = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                if (is_I_up(a[tid], y[tid], Cp, Cn))
                    if (f[tid] < up_value) {
                        up_value = f[tid];
                        i = tid;
                    }
            }
            for (int tid = 0; tid < ws_size; ++tid) {
                kIwsI[tid] = k_mat_rows[row_len * i + working_set[tid]];//K[i, wsi]
            }
            float_type low_value = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                if (is_I_low(a[tid], y[tid], Cp, Cn))
                    if (f[tid] > low_value) {
                        low_value = f[tid];
                    }
            }

//            printf("up = %lf, low = %lf\n", up_value, low_value);
            float_type local_diff = low_value - up_value;
            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                diff[0] = local_diff;
            }

            if (numOfIter > max_iter || local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(a[tid] - a_old[tid]) * y[tid];
                    alpha[wsi] = a[tid];
                }
                diff[1] = numOfIter;
                break;
            }
            int j2 = 0;
            float_type min_t = INFINITY;
            //select j2 using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                if (-up_value > -f[tid] && (is_I_low(a[tid], y[tid], Cp, Cn))) {
                    float_type aIJ = kd[i] + kd[tid] - 2 * kIwsI[tid];
                    float_type bIJ = -up_value + f[tid];
                    float_type ft = -bIJ * bIJ / aIJ;
                    if (ft < min_t) {
                        min_t = ft;
                        j2 = tid;
                    }
                }
            }

            //update alpha
//            if (tid == i)
            alpha_i_diff = y[i] > 0 ? Cp - a[i] : a[i];
//            if (tid == j2)
            alpha_j_diff = min(y[j2] > 0 ? a[j2] : Cn - a[j2],
                    (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            float_type l = min(alpha_i_diff, alpha_j_diff);

//            if (tid == i)
            a[i] += l * y[i];
//            if (tid == j2)
            a[j2] -= l * y[j2];


            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                float_type kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
        }
    }

    void c_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                     SyncArray<float_type> &alpha_diff,
                     const SyncArray<int> &working_set, float_type Cp, float_type Cn,
                     const SyncArray<kernel_type> &k_mat_rows,
                     const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps, SyncArray<float_type> &diff,
                     int max_iter) {
        c_smo_solve_kernel(y.host_data(), f_val.host_data(), alpha.host_data(), alpha_diff.host_data(),
                           working_set.host_data(), working_set.size(), Cp, Cn, k_mat_rows.host_data(),
                           k_mat_diag.host_data(), row_len, eps, diff.host_data(), max_iter);
    }

    void nu_smo_solve_kernel(const int *y, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                             const int *working_set, int ws_size, float_type C, const kernel_type *k_mat_rows,
                             const kernel_type *k_mat_diag, int row_len, float_type eps, float_type *diff,
                             int max_iter) {
        //allocate shared memory
        float_type alpha_i_diff; //delta alpha_i
        float_type alpha_j_diff;
        kernel_type *kd = new kernel_type[ws_size]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float_type *a_old = new float_type[ws_size];
        kernel_type *kIpwsI = new kernel_type[ws_size];
        kernel_type *kInwsI = new kernel_type[ws_size];
        float_type *f = new float_type[ws_size];
        for (int tid = 0; tid < ws_size; ++tid) {
            int wsi = working_set[tid];
            f[tid] = f_val[wsi];
            a_old[tid] = alpha[wsi];
            kd[tid] = k_mat_diag[wsi];
        }
        float_type local_eps;
        int numOfIter = 0;
        while (1) {
            //select I_up (y=+1)
            int ip = 0;
            float_type up_value_p = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] < C)
                    if (f[tid] < up_value_p) {
                        ip = tid;
                        up_value_p = f[tid];
                    }
            }

            for (int tid = 0; tid < ws_size; ++tid) {
                kIpwsI[tid] = k_mat_rows[row_len * ip + working_set[tid]];//K[i, wsi]
            }

            //select I_up (y=-1)
            int in = 0;
            float_type up_value_n = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] > 0)
                    if (f[tid] < up_value_n) {
                        in = tid;
                        up_value_n = f[tid];
                    }
            }
            for (int tid = 0; tid < ws_size; ++tid) {
                kInwsI[tid] = k_mat_rows[row_len * in + working_set[tid]];//K[i, wsi]
            }

            //select I_low (y=+1)
            float_type low_value_p = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] > 0)
                    if (f[tid] > low_value_p) {
                        low_value_p = f[tid];
                    }
            }


            //select I_low (y=-1)
            float_type low_value_n = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] < C)
                    if (f[tid] > low_value_n) {
                        low_value_n = f[tid];
                    }
            }

            float_type local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);

            if (numOfIter == 0) {
                local_eps = max(eps, 0.1 * local_diff);
                diff[0] = local_diff;
            }

            if (numOfIter > max_iter || local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                break;
            }

            //select j2p using second order heuristic
            int j2p = 0;
            float_type f_val_j2p = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_p > -f[tid] && y[wsi] > 0 && alpha[wsi] > 0) {
                    float_type aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float_type bIJ = -up_value_p + f[tid];
                    float_type f_t1 = -bIJ * bIJ / aIJ;
                    if (f_t1 < f_val_j2p) {
                        j2p = tid;
                        f_val_j2p = f_t1;
                    }
                }
            }

            //select j2n using second order heuristic
            int j2n = 0;
            float_type f_val_j2n = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_n > -f[tid] && y[wsi] < 0 && alpha[wsi] < C) {
                    float_type aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float_type bIJ = -up_value_n + f[tid];
                    float_type f_t2 = -bIJ * bIJ / aIJ;
                    if (f_t2 < f_val_j2n) {
                        j2n = tid;
                        f_val_j2n = f_t2;
                    }
                }
            }

            int i, j2;
            float_type up_value;
            kernel_type *kIwsI;
            if (f_val_j2p < f_val_j2n) {
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
            alpha_i_diff = y[working_set[i]] > 0 ? C - alpha[working_set[i]] : alpha[working_set[i]];
//            if (tid == j2)
            alpha_j_diff = min(y[working_set[j2]] > 0 ? alpha[working_set[j2]] : C - alpha[working_set[j2]],
                               (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            float_type l = min(alpha_i_diff, alpha_j_diff);

            alpha[working_set[i]] += l * y[working_set[i]];
            alpha[working_set[j2]] -= l * y[working_set[j2]];

            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                kernel_type kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            }
            numOfIter++;
        }
        delete[] kd;
        delete[] a_old;
        delete[] f;
        delete[] kIpwsI;
        delete[] kInwsI;
    }

    void nu_smo_solve(const SyncArray<int> &y, SyncArray<float_type> &f_val, SyncArray<float_type> &alpha,
                      SyncArray<float_type> &alpha_diff,
                      const SyncArray<int> &working_set, float_type C, const SyncArray<kernel_type> &k_mat_rows,
                      const SyncArray<kernel_type> &k_mat_diag, int row_len, float_type eps,
                      SyncArray<float_type> &diff,
                      int max_iter) {
        nu_smo_solve_kernel(y.host_data(), f_val.host_data(), alpha.host_data(), alpha_diff.host_data(),
                            working_set.host_data(), working_set.size(), C, k_mat_rows.host_data(),
                            k_mat_diag.host_data(), row_len, eps, diff.host_data(), max_iter);
    }

    void
    update_f(SyncArray<float_type> &f, const SyncArray<float_type> &alpha_diff,
             const SyncArray<kernel_type> &k_mat_rows,
             int n_instances) {
        //"n_instances" equals to the number of rows of the whole kernel matrix for both SVC and SVR.
        float_type *f_data = f.host_data();
        const float_type *alpha_diff_data = alpha_diff.host_data();
        const kernel_type *k_mat_rows_data = k_mat_rows.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; ++idx) {
            double sum_diff = 0;
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

