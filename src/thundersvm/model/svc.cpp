//
// Created by jiashuai on 17-9-21.
//
#include <iostream>
#include <iomanip>
#include <thundersvm/kernel/smo_kernel.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/solver/csmosolver.h>

using std::ofstream;
using std::endl;
using std::setprecision;
using std::ifstream;
using std::stringstream;

void SVC::model_setup(const DataSet &dataset, SvmParam &param) {

    //group instances with same class
    n_classes = dataset.n_classes();
    LOG(INFO) << "#classes = " << n_classes;
    this->label = dataset.label();
    SvmModel::model_setup(dataset, param);
    this->param.svm_type = SvmParam::C_SVC;
    //calculate class weight for each class
    c_weight = vector<float_type>(n_classes, 1);
    for (int i = 0; i < param.nr_weight; ++i) {
        bool found = false;
        for (int j = 0; j < n_classes; ++j) {
            if (param.weight_label[i] == dataset.label()[j]) {
                found = true;
                c_weight[j] *= param.weight[i];
                break;
            }
        }
        if (!found)
            LOG(WARNING) << "weighted label " << param.weight_label[i] << " not found";
    }

}

void SVC::train(const DataSet &dataset, SvmParam param) {
    DataSet dataset_ = dataset;
    dataset_.group_classes();
    model_setup(dataset_, param);

    vector<SyncArray<float_type>> alpha(n_binary_models);
    vector<bool> is_sv(dataset_.n_instances(), false);

    int k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            train_binary(dataset_, i, j, alpha[k], rho.host_data()[k]);
            vector<int> original_index = dataset_.original_index(i, j);
            CHECK_EQ(original_index.size(), alpha[k].size());
            const float_type *alpha_data = alpha[k].host_data();
            for (int l = 0; l < alpha[k].size(); ++l) {
                is_sv[original_index[l]] = is_sv[original_index[l]] || (alpha_data[l] != 0);
            }
            k++;
        }
    }

    for (int i = 0; i < dataset_.n_classes(); ++i) {
        vector<int> original_index = dataset_.original_index(i);
        DataSet::node2d i_instances = dataset_.instances(i);
        int *n_sv_data = n_sv.host_data();
        for (int j = 0; j < i_instances.size(); ++j) {
            if (is_sv[original_index[j]]) {
                n_sv_data[i]++;
                sv.push_back(i_instances[j]);
            }
        }
    }

    n_total_sv = sv.size();
    LOG(INFO) << "#total unique sv = " << n_total_sv;
    coef.resize((n_classes - 1) * n_total_sv);

    vector<int> sv_start(1, 0);
    const int *n_sv_data = n_sv.host_data();
    for (int i = 1; i < n_classes; ++i) {
        sv_start.push_back(sv_start[i - 1] + n_sv_data[i - 1]);
    }

    k = 0;
    float_type *coef_data = coef.host_data();
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            const float_type *alpha_data = alpha[k].host_data();
            vector<int> original_index = dataset_.original_index(i, j);
            int ci = dataset_.count()[i];
            int cj = dataset_.count()[j];
            int m = sv_start[i];
            for (int l = 0; l < ci; ++l) {
                if (is_sv[original_index[l]]) {
                    coef_data[(j - 1) * n_total_sv + m++] = alpha_data[l];
                }
            }
            m = sv_start[j];
            for (int l = ci; l < ci + cj; ++l) {
                if (is_sv[original_index[l]]) {
                    coef_data[i * n_total_sv + m++] = alpha_data[l];
                }
            }
            k++;
        }
    }

    ///TODO: Use coef instead of alpha_data to compute linear_coef_data
    if(param.kernel_type == SvmParam::LINEAR){
        int k = 0;
        linear_coef.resize(n_binary_models * dataset_.n_features());
        float_type *linear_coef_data = linear_coef.host_data();
        for (int i = 0; i < n_classes; i++){
            for (int j = i + 1; j < n_classes; j++){
                const float_type *alpha_data = alpha[k].host_data();
                DataSet::node2d ins = dataset_.instances(i, j);//get instances of class i and j
                for(int iid = 0; iid < ins.size(); iid++) {
                    for (int fid = 0; fid < ins[iid].size(); fid++) {
                        if(alpha_data[iid] != 0)
                            linear_coef_data[k * dataset_.n_features() + ins[iid][fid].index - 1] += alpha_data[iid] * ins[iid][fid].value;
                    }
                }
                k++;
            }
        }
    }

    //train probability
    if (1 == param.probability) {
        LOG(INFO) << "performing probability train";
        probA.resize(n_binary_models);
        probB.resize(n_binary_models);
        probability_train(dataset_);
    }

}

void SVC::train_binary(const DataSet &dataset, int i, int j, SyncArray<float_type> &alpha, float_type &rho) {
    DataSet::node2d ins = dataset.instances(i, j);//get instances of class i and j
    SyncArray<int> y(ins.size());
    alpha.resize(ins.size());
    SyncArray<float_type> f_val(ins.size());
    alpha.mem_set(0);
    int *y_data = y.host_data();
    float_type *f_val_data = f_val.host_data();
    for (int l = 0; l < dataset.count()[i]; ++l) {
        y_data[l] = +1;
        f_val_data[l] = -1;
    }
    for (int l = 0; l < dataset.count()[j]; ++l) {
        y_data[dataset.count()[i] + l] = -1;
        f_val_data[dataset.count()[i] + l] = +1;
    }
    KernelMatrix k_mat(ins, param);
    int ws_size = get_working_set_size(ins.size(), k_mat.n_features());
    CSMOSolver solver;
    solver.solve(k_mat, y, alpha, rho, f_val, param.epsilon, param.C * c_weight[i], param.C * c_weight[j], ws_size,
                 max_iter);
    LOG(INFO) << "rho = " << rho;
    int n_sv = 0;
    y_data = y.host_data();
    float_type *alpha_data = alpha.host_data();
    for (int l = 0; l < alpha.size(); ++l) {
        alpha_data[l] *= y_data[l];
        if (alpha_data[l] != 0) n_sv++;
    }
    LOG(INFO) << "#sv = " << n_sv;
}

vector<float_type> SVC::predict(const DataSet::node2d &instances, int batch_size) {
    dec_values.resize(instances.size() * n_binary_models);
    predict_dec_values(instances, dec_values, batch_size);
    return predict_label(dec_values, instances.size());
}

float_type sigmoidPredict(float_type dec_value, float_type A, float_type B) {
    double fApB = dec_value * A + B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return exp(-fApB) / (1.0 + exp(-fApB));
    else
        return 1.0 / (1 + exp(fApB));
}

void SVC::multiclass_probability(const vector<vector<float_type> > &r, vector<float_type> &p) const {
    int nrClass = n_classes;
    int t, j;
    int iter = 0, max_iter = max(100, nrClass);
    double **Q = (double **) malloc(sizeof(double *) * nrClass);
    double *Qp = (double *) malloc(sizeof(double) * nrClass);
    double pQp, eps = 0.005 / nrClass;

    for (t = 0; t < nrClass; t++) {
        p[t] = 1.0 / nrClass;  // Valid if k = 1
        Q[t] = (double *) malloc(sizeof(double) * nrClass);
        Q[t][t] = 0;
        for (j = 0; j < t; j++) {
            Q[t][t] += r[j][t] * r[j][t];
            Q[t][j] = Q[j][t];
        }
        for (j = t + 1; j < nrClass; j++) {
            Q[t][t] += r[j][t] * r[j][t];
            Q[t][j] = -r[j][t] * r[t][j];
        }
    }
    for (iter = 0; iter < max_iter; iter++) {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp = 0;
        for (t = 0; t < nrClass; t++) {
            Qp[t] = 0;
            for (j = 0; j < nrClass; j++)
                Qp[t] += Q[t][j] * p[j];
            pQp += p[t] * Qp[t];
        }
        double max_error = 0;
        for (t = 0; t < nrClass; t++) {
            double error = fabs(Qp[t] - pQp);
            if (error > max_error)
                max_error = error;
        }
        if (max_error < eps)
            break;

        for (t = 0; t < nrClass; t++) {
            double diff = (-Qp[t] + pQp) / Q[t][t];
            p[t] += diff;
            pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff)
                  / (1 + diff);
            for (j = 0; j < nrClass; j++) {
                Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
                p[j] /= (1 + diff);
            }
        }
    }
    if (iter >= max_iter)
        printf("Exceeds max_iter in multiclass_prob\n");
    for (t = 0; t < nrClass; t++)
        free(Q[t]);
    free(Q);
    free(Qp);
}

vector<float_type> SVC::predict_label(const SyncArray<float_type> &dec_values, int n_instances) {
    vector<float_type> predict_y;
    const float_type *dec_values_data = dec_values.host_data();
    if (0 == param.probability) {
        //predict y by voting among k(k-1)/2 models
        for (int l = 0; l < n_instances; ++l) {
            vector<int> votes(n_classes, 0);
            int k = 0;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    if (dec_values_data[l * n_binary_models + k] > 0)
                        votes[i]++;
                    else
                        votes[j]++;
                    k++;
                }
            }
            int maxVoteClass = 0;
            for (int i = 1; i < n_classes; ++i) {
                if (votes[i] > votes[maxVoteClass])
                    maxVoteClass = i;
            }
            predict_y.push_back((float) this->label[maxVoteClass]);
        }
    } else {
        LOG(INFO) << "predict with probability";
        this->prob_predict.clear();
        for (int l = 0; l < n_instances; ++l) {
            vector<vector<float_type> > r(n_classes, vector<float_type>(n_classes));
            float_type min_prob = 1e-7;
            int k = 0;
            for (int i = 0; i < n_classes; i++)
                for (int j = i + 1; j < n_classes; j++) {
                    r[i][j] = min(
                            max(sigmoidPredict(dec_values_data[l * n_binary_models + k], probA[k], probB[k]), min_prob),
                            1 - min_prob);
                    r[j][i] = 1 - r[i][j];
                    k++;
                }
            vector<float_type> p(n_classes);
            multiclass_probability(r, p);
            this->prob_predict.insert(prob_predict.end(), p.begin(), p.end());
            int max_prob_class = 0;
            for (int j = 0; j < n_classes; ++j) {
                if (p[j] > p[max_prob_class])
                    max_prob_class = j;
            }
            predict_y.push_back((float) this->label[max_prob_class]);
        }
    }
    return predict_y;
}

void sigmoidTrain(const float_type *decValues, const int l, const vector<int> &labels, float_type &A,
                  float_type &B) {
    double prior1 = 0, prior0 = 0;
    int i;

    for (i = 0; i < l; i++)
        if (labels[i] > 0)
            prior1 += 1;
        else
            prior0 += 1;

    int max_iter = 100;    // Maximal number of iterations
    double min_step = 1e-10;    // Minimal step taken in line search
    double sigma = 1e-12;    // For numerically strict PD of Hessian
    double eps = 1e-5;
    double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
    double loTarget = 1 / (prior0 + 2.0);
    double *t = (double *) malloc(sizeof(double) * l);
    double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
    double newA, newB, newf, d1, d2;
    int iter;

    // Initial Point and Initial Fun Value
    A = 0.0;
    B = log((prior0 + 1.0) / (prior1 + 1.0));
    double fval = 0.0;

    for (i = 0; i < l; i++) {
        if (labels[i] > 0)
            t[i] = hiTarget;
        else
            t[i] = loTarget;
        fApB = decValues[i] * A + B;
        if (fApB >= 0)
            fval += t[i] * fApB + log(1 + exp(-fApB));
        else
            fval += (t[i] - 1) * fApB + log(1 + exp(fApB));
    }
    for (iter = 0; iter < max_iter; iter++) {
        // Update Gradient and Hessian (use H' = H + sigma I)
        h11 = sigma; // numerically ensures strict PD
        h22 = sigma;
        h21 = 0.0;
        g1 = 0.0;
        g2 = 0.0;
        for (i = 0; i < l; i++) {
            fApB = decValues[i] * A + B;
            if (fApB >= 0) {
                p = exp(-fApB) / (1.0 + exp(-fApB));
                q = 1.0 / (1.0 + exp(-fApB));
            } else {
                p = 1.0 / (1.0 + exp(fApB));
                q = exp(fApB) / (1.0 + exp(fApB));
            }
            d2 = p * q;
            h11 += decValues[i] * decValues[i] * d2;
            h22 += d2;
            h21 += decValues[i] * d2;
            d1 = t[i] - p;
            g1 += decValues[i] * d1;
            g2 += d1;
        }

        // Stopping Criteria
        if (fabs(g1) < eps && fabs(g2) < eps)
            break;

        // Finding Newton direction: -inv(H') * g
        det = h11 * h22 - h21 * h21;
        dA = -(h22 * g1 - h21 * g2) / det;
        dB = -(-h21 * g1 + h11 * g2) / det;
        gd = g1 * dA + g2 * dB;

        stepsize = 1;        // Line Search
        while (stepsize >= min_step) {
            newA = A + stepsize * dA;
            newB = B + stepsize * dB;

            // New function value
            newf = 0.0;
            for (i = 0; i < l; i++) {
                fApB = decValues[i] * newA + newB;
                if (fApB >= 0)
                    newf += t[i] * fApB + log(1 + exp(-fApB));
                else
                    newf += (t[i] - 1) * fApB + log(1 + exp(fApB));
            }
            // Check sufficient decrease
            if (newf < fval + 0.0001 * stepsize * gd) {
                A = newA;
                B = newB;
                fval = newf;
                break;
            } else
                stepsize = stepsize / 2.0;
        }

        if (stepsize < min_step) {
            printf("Line search fails in two-class probability estimates\n");
            break;
        }
    }

    if (iter >= max_iter)
        printf(
                "Reaching maximal iterations in two-class probability estimates\n");
    free(t);
}

void SVC::probability_train(const DataSet &dataset) {
    SvmParam param_no_prob = param;
    param_no_prob.probability = 0;

    vector<float_type> dec_predict_all(dataset.n_instances() * n_binary_models);

    //cross-validation dec_values
    int n_fold = 5;
    for (int k = 0; k < n_fold; ++k) {
        SvmModel *temp_model = new SVC();
        DataSet::node2d x_train, x_test;
        vector<float_type> y_train, y_test;
        vector<int> test_idx;
        for (int i = 0; i < dataset.n_classes(); ++i) {
            int fold_test_count = dataset.count()[i] / n_fold;
            vector<int> class_idx = dataset.original_index(i);
            auto idx_begin = class_idx.begin() + fold_test_count * k;
            auto idx_end = idx_begin;
            if (k == n_fold - 1) {
                idx_end = class_idx.end();
            } else {
                while (idx_end != class_idx.end() && idx_end - idx_begin < fold_test_count) idx_end++;
            }
            for (int j: vector<int>(idx_begin, idx_end)) {
                x_test.push_back(dataset.instances()[j]);
                y_test.push_back(dataset.y()[j]);
                test_idx.push_back(j);
            }
            class_idx.erase(idx_begin, idx_end);
            for (int j:class_idx) {
                x_train.push_back(dataset.instances()[j]);
                y_train.push_back(dataset.y()[j]);
            }
        }
        DataSet train_dataset(x_train, dataset.n_features(), y_train);
        temp_model->train(train_dataset, param_no_prob);
        SyncArray<float_type> dec_predict(x_test.size() * n_binary_models);
        temp_model->predict_dec_values(x_test, dec_predict, 1000);
        float_type *dec_predict_data = dec_predict.host_data();
        for (int i = 0; i < x_test.size(); ++i) {
            memcpy(&dec_predict_all[test_idx[i] * n_binary_models], &dec_predict_data[i * n_binary_models],
                   sizeof(float_type) * n_binary_models);
        }
        delete temp_model;
    }
    int k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            vector<int> ori_idx;
            vector<int> y;
            vector<float_type> dec_values_subproblem;
            ori_idx = dataset.original_index(i);
            for (int l = 0; l < dataset.count()[i]; ++l) {
                y.push_back(+1);
                dec_values_subproblem.push_back(dec_predict_all[ori_idx[l] * n_binary_models + k]);
            }
            ori_idx = dataset.original_index(j);
            for (int l = 0; l < dataset.count()[j]; ++l) {
                y.push_back(-1);
                dec_values_subproblem.push_back(dec_predict_all[ori_idx[l] * n_binary_models + k]);
            }
            sigmoidTrain(dec_values_subproblem.data(), dec_values_subproblem.size(), y, probA[k], probB[k]);
            k++;
        }
    }
}

