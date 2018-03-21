
//
// Created by jiashuai on 17-11-7.
//


#include <thundersvm/kernel/smo_kernel.h>
#include <omp.h>
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

    void c_smo_solve_kernel(const int *y, float_type *f_val, float_type *alpha, float_type *alpha_diff,
                            const int *working_set,
                            int ws_size,
                            float Cp, float Cn, const float *k_mat_rows, const float *k_mat_diag, int row_len,
                            float_type eps,
                            float_type *diff, int max_iter) {
        //allocate shared memory
        float *f_val2reduce = new float[ws_size]; //f values used for reduction.
        float alpha_i_diff; //delta alpha_i
        float alpha_j_diff;
        float *kd = new float[ws_size]; // diagonal elements for kernel matrix

        //index, f value and alpha for each instance
        float *a_old = new float[ws_size];
        float *kIwsI = new float[ws_size];
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
            //select fUp and fLow
            int i = 0;
            float up_value = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (is_I_up(alpha[wsi], y[wsi], Cp, Cn))
                    if(f[tid] < up_value){
                        up_value = f[tid];
                        i = tid;
                    }
            }
            //int i = get_min_idx(f_val2reduce, ws_size);
            //float up_value = f_val2reduce[i];
            for (int tid = 0; tid < ws_size; ++tid) {
                kIwsI[tid] = k_mat_rows[row_len * i + working_set[tid]];//K[i, wsi]
            }
            int j1 = 0;
            float low_value = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (is_I_low(alpha[wsi], y[wsi], Cp, Cn))
                    if(f[tid] > low_value){
                        low_value = f[tid];
                        j1 = tid;
                    }
            }
            //int j1 = get_min_idx(f_val2reduce, ws_size);
            //float low_value = -f_val2reduce[j1];

            float local_diff = low_value - up_value;
            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                if(ws_size == row_len)//in this case, local smo = global smo
                    local_eps = eps;
                diff[0] = local_diff;
                if(diff[0] < eps){
                    diff[1] = numOfIter;
                    break;
                }
            }

            if (numOfIter > max_iter || local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                diff[1] = numOfIter;
/*
		std::cout<<"local f"<<std::endl;
                for(int i = 0; i < ws_size; i++)
                    std::cout<<f[i]<<" ";
                std::cout<<std::endl;
*/
                break;
            }
            int j2 = 0;
            float min_t = INFINITY;
            //select j2 using second order heuristic
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value > -f[tid] && (is_I_low(alpha[wsi], y[wsi], Cp, Cn))) {
                    float aIJ = kd[i] + kd[tid] - 2 * kIwsI[tid];
                    float bIJ = -up_value + f[tid];
                    float ft = -bIJ * bIJ / aIJ;
                    if(ft < min_t){
                        min_t = ft;
                        j2 = tid;
                    }
                }
            }
            //int j2 = get_min_idx(f_val2reduce, ws_size);

            //update alpha
//            if (tid == i)
            alpha_i_diff = y[working_set[i]] > 0 ? Cp - alpha[working_set[i]] : alpha[working_set[i]];
//            if (tid == j2)
            alpha_j_diff = min(y[working_set[j2]] > 0 ? alpha[working_set[j2]] : Cn - alpha[working_set[j2]],
                                (-up_value + f[j2]) / (kd[i] + kd[j2] - 2 * kIwsI[j2]));
            float l = min(alpha_i_diff, alpha_j_diff);

//            if (tid == i)
            alpha[working_set[i]] += l * y[working_set[i]];
//            if (tid == j2)
            alpha[working_set[j2]] -= l * y[working_set[j2]];


            //update f
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                float kJ2wsI = k_mat_rows[row_len * j2 + wsi];//K[J2, wsi]
                f[tid] -= l * (kJ2wsI - kIwsI[tid]);
            	if(ws_size == row_len){
			f_val[working_set[tid]] = f[tid];
//			std::cout<<"write back f"<<std::endl;
		} 
	    }
            numOfIter++;
/*
            if (numOfIter > max_iter){
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                diff[1] = numOfIter;
                break;
            }
*/
        }
        delete[] a_old;
        delete[] f;
        delete[] kIwsI;
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
            int ip = 0;
            float up_value_p = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] < C)
                    if(f[tid] < up_value_p){
                        ip = tid;
                        up_value_p = f[tid];
                    }
            }

            for (int tid = 0; tid < ws_size; ++tid) {
                kIpwsI[tid] = k_mat_rows[row_len * ip + working_set[tid]];//K[i, wsi]
            }

            //select I_up (y=-1)
            int in = 0;
            float up_value_n = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] > 0)
                    if(f[tid] < up_value_n){
                        in = tid;
                        up_value_n = f[tid];
                    }
            }
            for (int tid = 0; tid < ws_size; ++tid) {
                kInwsI[tid] = k_mat_rows[row_len * in + working_set[tid]];//K[i, wsi]
            }

            //select I_low (y=+1)
            int j1p = 0;
            float low_value_p = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] > 0 && alpha[wsi] > 0)
                    if(f[tid] > low_value_p){
                        j1p = tid;
                        low_value_p = f[tid];
                    }
            }


            //select I_low (y=-1)
            int j1n = 0;
            float low_value_n = -INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (y[wsi] < 0 && alpha[wsi] < C)
                    if(f[tid] > low_value_n){
                        j1n = tid;
                        low_value_n = f[tid];
                    }
            }

            float local_diff = max(low_value_p - up_value_p, low_value_n - up_value_n);

            if (numOfIter == 0) {
                local_eps = max(eps, 0.1f * local_diff);
                diff[0] = local_diff;
            }

            if (local_diff < local_eps) {
                for (int tid = 0; tid < ws_size; ++tid) {
                    int wsi = working_set[tid];
                    alpha_diff[tid] = -(alpha[wsi] - a_old[tid]) * y[wsi];
                }
                break;
            }

            //select j2p using second order heuristic
            int j2p = 0;
	    float f_val_j2p = INFINITY;
	    for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_p > -f[tid] && y[wsi] > 0 && alpha[wsi] > 0) {
                    float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float bIJ = -up_value_p + f[tid];
                    float f_t1  = -bIJ * bIJ / aIJ;
		    if(f_t1 < f_val_j2p){
			j2p = tid;
			f_val_j2p = f_t1;
		    }
                }
            }

            //select j2n using second order heuristic
            int j2n = 0;
            float f_val_j2n = INFINITY;
            for (int tid = 0; tid < ws_size; ++tid) {
                int wsi = working_set[tid];
                if (-up_value_n > -f[tid] && y[wsi] < 0 && alpha[wsi] < C) {
                    float aIJ = kd[ip] + kd[tid] - 2 * kIpwsI[tid];
                    float bIJ = -up_value_n + f[tid];
                    float f_t2 = -bIJ * bIJ / aIJ;
                    if(f_t2 < f_val_j2n){
                        j2n = tid;
                        f_val_j2n = f_t2;
                    }
                }
            }

            int i, j2;
            float up_value;
            float *kIwsI;
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
    			//double sum_diff = 0;
            for (int i = 0; i < alpha_diff.size(); ++i) {
                float_type d = alpha_diff_data[i];
                if (d != 0) {
                    //sum_diff += d * k_mat_rows_data[i * n_instances + idx];
                	f_data[idx] -= d * k_mat_rows_data[i * n_instances + idx];
		}
            }
            //f_data[idx] -= sum_diff;
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

