//
// Created by jiashuai on 17-9-21.
//
#include <iostream>
#include <iomanip>
#include <thundersvm/kernel/smo_kernel.h>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/solver/csmosolver.h>
#include "thrust/sort.h"

using std::ofstream;
using std::endl;
using std::setprecision;
using std::ifstream;
using std::stringstream;

void SVC::model_setup(const DataSet &dataset, SvmParam &param) {

    //group instances with same class
    n_classes = dataset.n_classes();
    this->label = dataset.label();
    SvmModel::model_setup(dataset, param);

    //calculate class weight for each class
    c_weight = vector<real>(n_classes, 1);
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

    vector<SyncData<real>> alpha(n_binary_models);
    vector<bool> is_sv(dataset_.n_instances(), false);

    int k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            train_binary(dataset_, i, j, alpha[k], rho[k]);
            vector<int> original_index = dataset_.original_index(i, j);
            CHECK_EQ(original_index.size(), alpha[k].size());
            for (int l = 0; l < alpha[k].size(); ++l) {
                is_sv[original_index[l]] = is_sv[original_index[l]] || (alpha[k][l] != 0);
            }
            k++;
        }
    }

    for (int i = 0; i < dataset_.n_classes(); ++i) {
        vector<int> original_index = dataset_.original_index(i);
        DataSet::node2d i_instances = dataset_.instances(i);
        for (int j = 0; j < i_instances.size(); ++j) {
            if (is_sv[original_index[j]]){
                n_sv[i]++;
                sv.push_back(i_instances[j]);
            }
        }
    }

    n_total_sv = sv.size();
    LOG(INFO)<<"#total unique sv = "<<n_total_sv;
    coef.resize((n_classes - 1) * n_total_sv);

    vector<int> sv_start(1,0);
    for (int i = 1; i < n_classes; ++i) {
        sv_start.push_back(sv_start[i-1] + n_sv[i-1]);
    }

    k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i+1; j < n_classes; ++j) {
            vector<int> original_index = dataset_.original_index(i,j);
            int ci = dataset_.count()[i];
            int cj = dataset_.count()[j];
            int m = sv_start[i];
            for (int l = 0; l < ci; ++l) {
                if (is_sv[original_index[l]]){
                    coef[(j-1) * n_total_sv + m++] = alpha[k][l];
                }
            }
            m = sv_start[j];
            for (int l = ci; l < ci + cj; ++l) {
                if (is_sv[original_index[l]]){
                    coef[i * n_total_sv + m++] = alpha[k][l];
                }
            }
            k++;
        }
    }

    //train probability
    if (1 == param.probability) {
        LOG(INFO) << "performing probability train";
        probA.resize(n_binary_models);
        probB.resize(n_binary_models);
        probability_train(dataset);
    }
}

void SVC::train_binary(const DataSet &dataset, int i, int j, SyncData<real> &alpha, real &rho) {
    DataSet::node2d ins = dataset.instances(i, j);//get instances of class i and j
    SyncData<int> y(ins.size());
    alpha.resize(ins.size());
    SyncData<real> f_val(ins.size());
    alpha.mem_set(0);
    for (int l = 0; l < dataset.count()[i]; ++l) {
        y[l] = +1;
        f_val[l] = -1;
    }
    for (int l = 0; l < dataset.count()[j]; ++l) {
        y[dataset.count()[i] + l] = -1;
        f_val[dataset.count()[i] + l] = +1;
    }
    KernelMatrix k_mat(ins, param);
    int ws_size = min(min(max2power(dataset.count()[i]), max2power(dataset.count()[j])) * 2, 1024);
    CSMOSolver solver;
    solver.solve(k_mat, y, alpha, rho, f_val, param.epsilon, param.C * c_weight[i], param.C * c_weight[j], ws_size);
    LOG(INFO)<<"rho = "<<rho;
    int n_sv = 0;
    for (int l = 0; l < alpha.size(); ++l) {
        alpha[l] *= y[l];
        if (alpha[l] != 0) n_sv++;
    }
    LOG(INFO)<<"#sv = "<<n_sv;
}

vector<real> SVC::predict(const DataSet::node2d &instances, int batch_size) {
    SyncData<real> dec_values(instances.size() * n_binary_models);
    predict_dec_values(instances, dec_values, batch_size);
    return predict_label(dec_values, instances.size());
}

real sigmoidPredict(real dec_value, real A, real B) {
    double fApB = dec_value * A + B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return exp(-fApB) / (1.0 + exp(-fApB));
    else
        return 1.0 / (1 + exp(fApB));
}

void SVC::multiclass_probability(const vector<vector<real> > &r, vector<real> &p) const {
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

vector<real> SVC::predict_label(const SyncData<real> &dec_values, int n_instances) const {
    vector<real> predict_y;
    if (0 == param.probability) {
        //predict y by voting among k(k-1)/2 models
        for (int l = 0; l < n_instances; ++l) {
            vector<int> votes(n_binary_models, 0);
            int k = 0;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    if (dec_values[l * n_binary_models + k] > 0)
                        votes[i]++;
                    else
                        votes[j]++;
                    k++;
                }
            }
            int maxVoteClass = 0;
            for (int i = 0; i < n_classes; ++i) {
                if (votes[i] > votes[maxVoteClass])
                    maxVoteClass = i;
            }
            predict_y.push_back((float) this->label[maxVoteClass]);
        }
    } else {
        LOG(INFO) << "predict with probability";
        for (int l = 0; l < n_instances; ++l) {
            vector<vector<real> > r(n_classes, vector<real>(n_classes));
            double min_prob = 1e-7;
            int k = 0;
            for (int i = 0; i < n_classes; i++)
                for (int j = i + 1; j < n_classes; j++) {
                    r[i][j] = min(
                            max(sigmoidPredict(dec_values[l * n_binary_models + k], probA[k], probB[k]), min_prob),
                            1 - min_prob);
                    r[j][i] = 1 - r[i][j];
                    k++;
                }
            vector<real> p(n_classes);
            multiclass_probability(r, p);
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

void sigmoidTrain(const real *decValues, const int l, const vector<int> &labels, real &A,
                  real &B) {
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
    SyncData<real> dec_values(dataset.n_instances() * n_binary_models);
    predict_dec_values(dataset.instances(), dec_values, 10000);
    int k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            vector<int> ori_idx;
            vector<int> y;
            vector<real> dec_values_subproblem;
            ori_idx = dataset.original_index(i);
            for (int l = 0; l < dataset.count()[i]; ++l) {
                y.push_back(+1);
                dec_values_subproblem.push_back(dec_values[ori_idx[l] * n_binary_models + k]);
            }
            ori_idx = dataset.original_index(j);
            for (int l = 0; l < dataset.count()[j]; ++l) {
                y.push_back(-1);
                dec_values_subproblem.push_back(dec_values[ori_idx[l] * n_binary_models + k]);
            }
            sigmoidTrain(dec_values_subproblem.data(), dec_values_subproblem.size(), y, probA[k], probB[k]);
            k++;
        }
    }
}

