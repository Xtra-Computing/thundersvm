/*
   * @author: created by ss on 16-11-2.
   * @brief: multi-class svm training, prediction, svm with probability output
   *
*/

#include "svmModel.h"

#include <cstdio>
#include "svmPredictor.h"
#include "../svm-shared/HessianIO/deviceHessian.h"
#include "../svm-shared/storageManager.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "trainingFunction.h"

unsigned int SvmModel::getK(int i, int j) const {
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void SvmModel::fit(const SvmProblem &problem, const SVMParam &param) {
    nrClass = problem.getNumOfClasses();
    cnr2 = (nrClass) * (nrClass - 1) / 2;
    coef.clear();
    rho.clear();
    probA.clear();
    probB.clear();
    supportVectors.clear();
    label.clear();
    probability = false;

    coef.resize(cnr2);
    rho.resize(cnr2);
    probA.resize(cnr2);
    probB.resize(cnr2);
    supportVectors.resize(cnr2);

    this->param = param;
    label = problem.label;
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
                //binary model has only one sub-model
                sigmoidTrain(decValues.front().data(), subProblem.getNumOfSamples(), subProblem.v_nLabels, probA[k],
                             probB[k]);
                probability = true;
            }
            svm_model binaryModel = trainBinarySVM(subProblem, param);
            addBinaryModel(subProblem, binaryModel, i, j);
            k++;
        }
    }
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
    for (int l = 0; l < bModel.nSV[0] + bModel.nSV[1]; ++l) {
        coef[k].push_back(bModel.sv_coef[0][l]);
        supportVectors[k].push_back(problem.v_vSamples[bModel.pnIndexofSV[l]]);
    }
    rho[k] = bModel.rho[0];
}

void
SvmModel::predictValues(const vector<vector<float_point> > &v_vSamples,
                        vector<vector<float_point> > &decisionValues) const {
    decisionValues.clear();
    for (int k = 0; k < cnr2; ++k) {
        float_point *devKernelValues;
        checkCudaErrors(cudaMalloc((void **) &devKernelValues,
                                   sizeof(float_point) * v_vSamples.size() * supportVectors[k].size()));
        computeKernelValuesOnGPU(v_vSamples, supportVectors[k], devKernelValues);
        vector<float_point> decValues41BinaryModel;
        predictLabels(devKernelValues, v_vSamples.size(), decValues41BinaryModel, k);
        decisionValues.push_back(decValues41BinaryModel);
        checkCudaErrors(cudaFree(devKernelValues));
    }
}

vector<int> SvmModel::predict(const vector<vector<float_point> > &v_vSamples, bool probability) const {
    vector<vector<float_point> > decisionValues;
    predictValues(v_vSamples, decisionValues);
    vector<int> labels;
    if (!probability) {
        for (int l = 0; l < v_vSamples.size(); ++l) {
            vector<int> votes(nrClass, 0);
            int k = 0;
            for (int i = 0; i < nrClass; ++i) {
                for (int j = i + 1; j < nrClass; ++j) {
                    if (decisionValues[k++][l] > 0)
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
                        max(sigmoidPredict(decValues[k][l], probA[k], probB[k]), min_prob), 1 - min_prob);
                r[j][i] = 1 - r[i][j];
                k++;
            }
        vector<float_point> p(nrClass);
        multiClassProbability(r, p);
        result.push_back(p);
    }
    return result;
}

//todo move this kernel function to a proper file
__global__ void
rbfKernel(float_point *samples, int numOfSamples, float_point *supportVectors, int numOfSVs, int numOfFeatures,
          float_point *kernelValues, float_point gamma) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row = idx / numOfSVs;
    int col = idx % numOfSVs;
    if (row < numOfSamples) {
        float_point *sample = samples + row * numOfFeatures;
        float_point *supportVector = supportVectors + col * numOfFeatures;
        float_point sum = 0;
        for (int i = 0; i < numOfFeatures; ++i) {
            float_point d = sample[i] - supportVector[i];
            sum += d * d;
        }
        kernelValues[idx] = exp(-gamma * sum);
    }
};

void SvmModel::computeKernelValuesOnGPU(const vector<vector<float_point> > &samples,
                                        const vector<vector<float_point> > &supportVectors,
                                        float_point *devKernelValues) const {
    //copy samples and support vectors to device
    int numOfFeatures = (int) samples.front().size();
    float_point *devSamples, *devSupportVectors;
    checkCudaErrors(cudaMalloc((void **) &devSamples, sizeof(float_point) * samples.size() * numOfFeatures));
    checkCudaErrors(
            cudaMalloc((void **) &devSupportVectors, sizeof(float_point) * supportVectors.size() * numOfFeatures));
    size_t vectorSize = sizeof(float_point) * numOfFeatures;
    for (int i = 0; i < samples.size(); ++i) {
        checkCudaErrors(
                cudaMemcpy(devSamples + i * numOfFeatures, samples[i].data(), vectorSize, cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < supportVectors.size(); ++i) {
        checkCudaErrors(cudaMemcpy(devSupportVectors + i * numOfFeatures, supportVectors[i].data(), vectorSize,
                                   cudaMemcpyHostToDevice));
    }

    //compute kernel values
    int numOfBlock = (int) (Ceil(samples.size() * supportVectors.size(), BLOCK_SIZE));
    rbfKernel << < numOfBlock, BLOCK_SIZE >> > (devSamples, samples.size(), devSupportVectors, supportVectors.size(),
            numOfFeatures, devKernelValues, param.gamma);

    checkCudaErrors(cudaFree(devSamples));
    checkCudaErrors(cudaFree(devSupportVectors));
}

void SvmModel::predictLabels(float_point *devKernelValues, int numOfSamples, vector<float_point> &classificationResult,
                             int k) const {
    //get information from SVM model
    int nNumofSVs = (int) supportVectors[k].size();
    classificationResult.resize(numOfSamples * nNumofSVs);
    float_point fBias = rho[k];
    const float_point *pfSVsYiAlpha = coef[k].data();
    float_point *pfDevSVsYiAlpha;
    checkCudaErrors(cudaMalloc((void **) &pfDevSVsYiAlpha, sizeof(float_point) * nNumofSVs));
    checkCudaErrors(
            cudaMemcpy(pfDevSVsYiAlpha, pfSVsYiAlpha, sizeof(float_point) * nNumofSVs, cudaMemcpyHostToDevice));
    //compute y_i*alpha_i*K(i, z)
    int nVecMatxMulGridDimY = numOfSamples;
    int nVecMatxMulGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
    dim3 vecMatxMulGridDim(nVecMatxMulGridDimX, nVecMatxMulGridDimY);
    VectorMatrixMul << < vecMatxMulGridDim, BLOCK_SIZE >> >
                                            (pfDevSVsYiAlpha, devKernelValues, numOfSamples, nNumofSVs);
    //perform classification
    ComputeClassLabel(numOfSamples, devKernelValues,
                      nNumofSVs, fBias, classificationResult.data());
    checkCudaErrors(cudaFree(pfDevSVsYiAlpha));
}

/*
 * @brief: compute/predict the labels of testing samples
 * @output: a set of class labels, associated to testing samples
 */
float_point *SvmModel::ComputeClassLabel(int nNumofTestSamples,
                                         float_point *pfDevSVYiAlphaHessian, const int &nNumofSVs,
                                         float_point fBias, float_point *pfFinalResult) const {
    float_point *pfReturn = NULL;
    if (nNumofTestSamples <= 0 ||
        pfDevSVYiAlphaHessian == NULL ||
        nNumofSVs <= 0) {
        cerr << "error in ComputeClassLabel: invalid input params" << endl;
        return pfReturn;
    }

    //compute the size of current processing testing samples
/*	size_t nFreeMemory,nTotalMemory;
	cudaMemGetInfo(&nFreeMemory,&nTotalMemory);
*/    int nMaxSizeofProcessingSample = ((CACHE_SIZE) * 1024 * 1024 * 4 / (sizeof(float_point) * nNumofSVs));

    //reduce by half
    nMaxSizeofProcessingSample = nMaxSizeofProcessingSample / 2;

    //if the number of samples in small
    if (nMaxSizeofProcessingSample > nNumofTestSamples) {
        nMaxSizeofProcessingSample = nNumofTestSamples;
    }
    //compute grid size, and block size for partial sum
    int nPartialGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
    int nPartialGridDimY = nMaxSizeofProcessingSample;
    dim3 dimPartialSumGrid(nPartialGridDimX, nPartialGridDimY);
    dim3 dimPartialSumBlock(BLOCK_SIZE);

    //compute grid size, and block size for global sum and class label computing
    int nGlobalGridDimX = 1;
    int nGlobalGridDimY = nMaxSizeofProcessingSample;
    dim3 dimGlobalSumGrid(nGlobalGridDimX, nGlobalGridDimY); //can use 1D grid
    dim3 dimGlobalSumBlock(nPartialGridDimX);

    //memory for computing partial sum by GPU
    float_point *pfDevPartialSum;
//	cout << "dimx=" << nPartialGridDimX << "; dimy=" << nPartialGridDimY << endl;
    checkCudaErrors(cudaMalloc((void **) &pfDevPartialSum, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));
    checkCudaErrors(cudaMemset(pfDevPartialSum, 0, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));

    //memory for computing global sum by GPU
    float_point *pfDevClassificationResult;
    checkCudaErrors(cudaMalloc((void **) &pfDevClassificationResult, sizeof(float_point) * nGlobalGridDimY));
    checkCudaErrors(cudaMemset(pfDevClassificationResult, 0, sizeof(float_point) * nGlobalGridDimY));

    //reduce step size of partial sum, and global sum
    int nPartialReduceStepSize = 0;
    nPartialReduceStepSize = (int) pow(2, (ceil(log2((float) BLOCK_SIZE)) - 1));
    int nGlobalReduceStepSize = 0;
    nGlobalReduceStepSize = (int) pow(2, ceil(log2((float) nPartialGridDimX)) - 1);

    for (int nStartPosofTestSample = 0;
         nStartPosofTestSample < nNumofTestSamples; nStartPosofTestSample += nMaxSizeofProcessingSample) {
        if (nStartPosofTestSample + nMaxSizeofProcessingSample > nNumofTestSamples) {
            //the last part of the testing samples
            nMaxSizeofProcessingSample = nNumofTestSamples - nStartPosofTestSample;
            nPartialGridDimY = nMaxSizeofProcessingSample;
            dimPartialSumGrid = dim3(nPartialGridDimX, nPartialGridDimY);
            nGlobalGridDimY = nMaxSizeofProcessingSample;
            dimGlobalSumGrid = dim3(nGlobalGridDimX, nGlobalGridDimY);

            checkCudaErrors(cudaFree(pfDevPartialSum));
            checkCudaErrors(
                    cudaMalloc((void **) &pfDevPartialSum, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));
            checkCudaErrors(cudaMemset(pfDevPartialSum, 0, sizeof(float_point) * nPartialGridDimX * nPartialGridDimY));

            checkCudaErrors(cudaFree(pfDevClassificationResult));
            checkCudaErrors(cudaMalloc((void **) &pfDevClassificationResult, sizeof(float_point) * nGlobalGridDimY));
            checkCudaErrors(cudaMemset(pfDevClassificationResult, 0, sizeof(float_point) * nGlobalGridDimY));
        }
        /********* compute partial sum **********/
        ComputeKernelPartialSum << < dimPartialSumGrid, dimPartialSumBlock, BLOCK_SIZE * sizeof(float_point) >> >
                                                                            (pfDevSVYiAlphaHessian, nNumofSVs, pfDevPartialSum,
                                                                                    nPartialReduceStepSize);
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cerr << "cuda error in ComputeClassLabel: failed at ComputePartialSum: " << cudaGetErrorString(error)
                 << endl;
            return pfReturn;
        }

        /********** compute global sum and class label *********/
        //compute global sum
        ComputeKernelGlobalSum << < dimGlobalSumGrid, dimGlobalSumBlock, nPartialGridDimX * sizeof(float_point) >> >
                                                                         (pfDevClassificationResult, fBias,
                                                                                 pfDevPartialSum, nGlobalReduceStepSize);
        cudaDeviceSynchronize();

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            cerr << "cuda error in ComputeClassLabel: failed at ComputeGlobalSum: " << cudaGetErrorString(error)
                 << endl;
            return pfReturn;
        }

        //copy classification result back
        checkCudaErrors(cudaMemcpy(pfFinalResult + nStartPosofTestSample, pfDevClassificationResult,
                                   nMaxSizeofProcessingSample * sizeof(float_point), cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaFree(pfDevPartialSum));
    checkCudaErrors(cudaFree(pfDevClassificationResult));

    pfReturn = pfFinalResult;
    return pfReturn;
}

bool SvmModel::isProbability() const {
    return probability;
}

