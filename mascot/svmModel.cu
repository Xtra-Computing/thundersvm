/*
   * @author: created by ss on 16-11-2.
   * @brief: multi-class svm training, prediction, svm with probability output
   *
*/

#include "svmModel.h"

#include "svmPredictor.h"
#include "../svm-shared/HessianIO/deviceHessian.h"
#include "../svm-shared/storageManager.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "trainingFunction.h"

//todo move these kernel functions to a proper file
__global__ void
rbfKernel(const float_point *samples, int numOfSamples, const float_point *supportVectors, int numOfSVs,
          int numOfFeatures,
          float_point *kernelValues, float_point gamma,
          const float_point *coef) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / numOfSVs;
    int SVId = idx % numOfSVs;
    if (sampleId < numOfSamples) {
        const float_point *sample = samples + sampleId * numOfFeatures;
        const float_point *supportVector = supportVectors + SVId * numOfFeatures;
        float_point sum = 0;
        for (int i = 0; i < numOfFeatures; ++i) {
            float_point d = sample[i] - supportVector[i];
            sum += d * d;
        }
        kernelValues[idx] = coef[SVId] * exp(-gamma * sum);
    }
};

__global__ void sumKernelValues(const float *kernelValues, int numOfSamples, int numOfSVs, int cnr2,
                                const int *start, const int *count,
                                const float *bias, float_point *decValues) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / cnr2;
    int modelId = idx % cnr2;
    if (sampleId < numOfSamples) {
        float_point sum = 0;
        const float_point *kernelValue = kernelValues + sampleId * numOfSVs + start[modelId];
        for (int i = 0; i < count[modelId]; ++i) {
            sum += kernelValue[i];
        }
        decValues[idx] = sum - bias[modelId];
    }
}

SvmModel::~SvmModel() {
    checkCudaErrors(cudaFree(devSVs));
    checkCudaErrors(cudaFree(devCoef));
    checkCudaErrors(cudaFree(devStart));
    checkCudaErrors(cudaFree(devCount));
    checkCudaErrors(cudaFree(devProbA));
    checkCudaErrors(cudaFree(devProbB));
    checkCudaErrors(cudaFree(devRho));
}

unsigned int SvmModel::getK(int i, int j) const {
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void SvmModel::fit(const SvmProblem &problem, const SVMParam &param) {
    //reset model to fit a new SvmProblem
    nrClass = problem.getNumOfClasses();
    cnr2 = (nrClass) * (nrClass - 1) / 2;
    numOfFeatures = problem.v_vSamples.front().size();
    numOfSVs = 0;
    coef.clear();
    rho.clear();
    probA.clear();
    probB.clear();
    supportVectors.clear();
    label.clear();
    start.clear();
    count.clear();
    probability = false;

    coef.resize(cnr2);
    rho.resize(cnr2);
    probA.resize(cnr2);
    probB.resize(cnr2);
    supportVectors.resize(cnr2);

    this->param = param;
    label = problem.label;

    //train nrClass*(nrClass-1)/2 binary models
    int k = 0;
    for (int i = 0; i < nrClass; ++i) {
        for (int j = i + 1; j < nrClass; ++j) {
            SvmProblem subProblem = problem.getSubProblem(i, j);
            printf("training classifier with label %d and %d\n", i, j);
            if (param.probability) {
                SVMParam probParam = param;
                probParam.probability = 0;
                probParam.C = 1.0;
                SvmModel model;
                model.fit(subProblem, probParam);
                vector<vector<float_point> > decValues;
                //todo predict with cross validation
                model.predictValues(subProblem.v_vSamples, decValues);
                for (int l = 1; l < subProblem.v_vSamples.size(); ++l) {
                    decValues[0].push_back(decValues[l][0]);
                }
                sigmoidTrain(decValues.front().data(), subProblem.getNumOfSamples(), subProblem.v_nLabels, probA[k],
                             probB[k]);
                probability = true;
            }
            svm_model binaryModel = trainBinarySVM(subProblem, param);
            addBinaryModel(subProblem, binaryModel, i, j);
            k++;
        }
    }
    int _start = 0;
    for (int i = 0; i < cnr2; ++i) {
        start.push_back(_start);
        count.push_back(supportVectors[i].size());
        _start += count[i];
    }
    transferToDevice();
}

void SvmModel::transferToDevice() {
    int svLength = numOfFeatures;
    checkCudaErrors(cudaMalloc((void **) &devSVs, sizeof(float_point) * numOfSVs * svLength));
    checkCudaErrors(cudaMalloc((void **) &devCoef, sizeof(float_point) * numOfSVs));
    checkCudaErrors(cudaMalloc((void **) &devStart, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devCount, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devProbA, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devProbB, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devRho, sizeof(float_point) * cnr2));
    for (int i = 0; i < cnr2; ++i) {
        float_point *sv4BinaryModel = new float_point[supportVectors[i].size() * svLength];
        for (int j = 0; j < supportVectors[i].size(); ++j) {
            memcpy(sv4BinaryModel + j * svLength, supportVectors[i][j].data(), sizeof(float_point) * svLength);
        }
        checkCudaErrors(cudaMemcpy(devSVs + start[i] * svLength, sv4BinaryModel,
                                   sizeof(float_point) * count[i] * svLength, cudaMemcpyHostToDevice));
        delete[] sv4BinaryModel;
        checkCudaErrors(cudaMemcpy(devCoef + start[i], coef[i].data(), sizeof(float_point) * count[i],
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(devProbA, probA.data(), sizeof(float_point) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devProbB, probB.data(), sizeof(float_point) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devStart, start.data(), sizeof(int) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devCount, count.data(), sizeof(int) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devRho, rho.data(), sizeof(float_point) * cnr2, cudaMemcpyHostToDevice));
}

void SvmModel::sigmoidTrain(const float_point *decValues, const int l, const vector<int> &labels, float_point &A,
                            float_point &B) {
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

void SvmModel::addBinaryModel(const SvmProblem &problem, const svm_model &bModel, int i, int j) {
    unsigned int k = getK(i, j);
    supportVectors[k].resize(bModel.nSV[0] + bModel.nSV[1]);
    for (int l = 0; l < bModel.nSV[0] + bModel.nSV[1]; ++l) {
        coef[k].push_back(bModel.sv_coef[0][l]);
        supportVectors[k][l] = problem.v_vSamples[bModel.pnIndexofSV[l]];
    }
    rho[k] = bModel.rho[0];
    numOfSVs += bModel.nSV[0] + bModel.nSV[1];
}

void
SvmModel::predictValues(const vector<vector<float_point> > &v_vSamples,
                        vector<vector<float_point> > &decisionValues) const {
    //copy samples to device
    float_point *devSamples;
    checkCudaErrors(cudaMalloc((void **) &devSamples, sizeof(float_point) * v_vSamples.size() * numOfFeatures));
    for (int i = 0; i < v_vSamples.size(); ++i) {
        checkCudaErrors(cudaMemcpy(devSamples + i * numOfFeatures, v_vSamples[i].data(),
                                   sizeof(float_point) * numOfFeatures, cudaMemcpyHostToDevice));
    }


    float_point *devKernelValues;
    checkCudaErrors(cudaMalloc((void **) &devKernelValues,
                               sizeof(float_point) * v_vSamples.size() * numOfSVs));
    int numOfBlock = Ceil(v_vSamples.size() * numOfSVs, BLOCK_SIZE);
    rbfKernel << < numOfBlock, BLOCK_SIZE >> > (devSamples, v_vSamples.size(),
            devSVs, numOfSVs, numOfFeatures, devKernelValues, param.gamma, devCoef);
    numOfBlock = Ceil(v_vSamples.size() * cnr2, BLOCK_SIZE);
    float_point *devDecisionValues;
    checkCudaErrors(cudaMalloc((void **) &devDecisionValues, sizeof(float_point) * v_vSamples.size() * cnr2));
    sumKernelValues << < numOfBlock, BLOCK_SIZE >> > (devKernelValues, v_vSamples.size(),
            numOfSVs, cnr2, devStart, devCount, devRho, devDecisionValues);
    float_point *tempDecValues = new float_point[v_vSamples.size() * cnr2];
    checkCudaErrors(cudaMemcpy(tempDecValues, devDecisionValues,
                               sizeof(float_point) * v_vSamples.size() * cnr2, cudaMemcpyDeviceToHost));
    decisionValues = vector<vector<float_point> >(v_vSamples.size(), vector<float_point>(cnr2));
    for (int i = 0; i < decisionValues.size(); ++i) {
        memcpy(decisionValues[i].data(), tempDecValues + i * cnr2, sizeof(float_point) * cnr2);
    }
    delete[] tempDecValues;
    checkCudaErrors(cudaFree(devSamples));
    checkCudaErrors(cudaFree(devDecisionValues));
    checkCudaErrors(cudaFree(devKernelValues));
}

vector<int> SvmModel::predict(const vector<vector<float_point> > &v_vSamples, bool probability) const {
    vector<int> labels;
    if (!probability) {
        vector<vector<float_point> > decisionValues;
        predictValues(v_vSamples, decisionValues);
        for (int l = 0; l < v_vSamples.size(); ++l) {
            vector<int> votes(nrClass, 0);
            int k = 0;
            for (int i = 0; i < nrClass; ++i) {
                for (int j = i + 1; j < nrClass; ++j) {
                    if (decisionValues[l][k++] > 0)
                        votes[i]++;
                    else
                        votes[j]++;
                }
            }
            int maxVoteClass = 0;
            for (int i = 0; i < nrClass; ++i) {
                if (votes[i] > votes[maxVoteClass])
                    maxVoteClass = i;
            }
            labels.push_back(this->label[maxVoteClass]);
        }
    } else {
        printf("predict with probability\n");
        assert(this->probability);
        vector<vector<float_point> > prob = predictProbability(v_vSamples);
        // todo select max using GPU
        for (int i = 0; i < v_vSamples.size(); ++i) {
            int maxProbClass = 0;
            for (int j = 0; j < nrClass; ++j) {
                if (prob[i][j] > prob[i][maxProbClass])
                    maxProbClass = j;
            }
            labels.push_back(this->label[maxProbClass]);
        }
    }
    return labels;
}

float_point SvmModel::sigmoidPredict(float_point decValue, float_point A, float_point B) const {
    double fApB = decValue * A + B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return exp(-fApB) / (1.0 + exp(-fApB));
    else
        return 1.0 / (1 + exp(fApB));
}

void SvmModel::multiClassProbability(const vector<vector<float_point> > &r, vector<float_point> &p) const {
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

vector<vector<float_point> > SvmModel::predictProbability(const vector<vector<float_point> > &v_vSamples) const {
    vector<vector<float_point> > result;
    vector<vector<float_point> > decValues;
    predictValues(v_vSamples, decValues);
    for (int l = 0; l < v_vSamples.size(); ++l) {
        vector<vector<float_point> > r(nrClass, vector<float_point>(nrClass));
        double min_prob = 1e-7;
        int k = 0;
        for (int i = 0; i < nrClass; i++)
            for (int j = i + 1; j < nrClass; j++) {
                r[i][j] = min(
                        max(sigmoidPredict(decValues[l][k], probA[k], probB[k]), min_prob), 1 - min_prob);
                r[j][i] = 1 - r[i][j];
                k++;
            }
        vector<float_point> p(nrClass);
        multiClassProbability(r, p);
        result.push_back(p);
    }
    return result;
}

bool SvmModel::isProbability() const {
    return probability;
}

