/*
   * @author: created by ss on 16-11-2.
   * @brief: multi-class svm training, prediction, svm with probability output 
   *
*/

#include <cstdio>
#include "svmModel.h"
#include "svmPredictor.h"
#include "../svm-shared/HessianIO/deviceHessian.h"
#include "../svm-shared/storageManager.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "trainingFunction.h"

unsigned int svmModel::getK(int i, int j) const {
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void svmModel::fit(const svmProblem &problem, const SVMParam &param) {
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
            svmProblem subProblem = problem.getSubProblem(i, j);
            printf("training classifier with label %d and %d\n", i, j);
            if (param.probability) {
                SVMParam probParam = param;
                probParam.probability = 0;
                probParam.C = 1.0;
                svmModel model;
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

void svmModel::sigmoidTrain(const float_point *decValues, const int l, const vector<int> &labels, float_point &A,
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

void svmModel::addBinaryModel(const svmProblem &problem, const svm_model &bModel, int i, int j) {
    unsigned int k = getK(i, j);
    for (int l = 0; l < bModel.nSV[0] + bModel.nSV[1]; ++l) {
//        printf("%d:%f.2|",bModel.pnIndexofSV[l],bModel.sv_coef[0][l]);
        coef[k].push_back(bModel.sv_coef[0][l]);
        supportVectors[k].push_back(problem.v_vSamples[bModel.pnIndexofSV[l]]);
//        printf("%d|",problem.v_nLabels[bModel.pnIndexofSV[l]]);
    }
//    printf("\n");
    rho[k] = bModel.rho[0];
}

void
svmModel::predictValues(const vector<vector<float_point> > &v_vSamples,
                        vector<vector<float_point> > &decisionValues) const {
    decisionValues.clear();
    for (int k = 0; k < cnr2; ++k) {
        vector<float_point> kernelValues;
        computeKernelValuesOnFly(v_vSamples, supportVectors[k], kernelValues);
        vector<float_point> decValues41BinaryModel;
        predictLabels(kernelValues, decValues41BinaryModel, k);
        decisionValues.push_back(decValues41BinaryModel);
    }
}

vector<int> svmModel::predict(const vector<vector<float_point> > &v_vSamples, bool probability) const {
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

float_point svmModel::sigmoidPredict(float_point decValue, float_point A, float_point B) const {
    double fApB = decValue * A + B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return exp(-fApB) / (1.0 + exp(-fApB));
    else
        return 1.0 / (1 + exp(fApB));
}

void svmModel::multiClassProbability(const vector<vector<float_point> > &r, vector<float_point> &p) const {
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

vector<vector<float_point> > svmModel::predictProbability(const vector<vector<float_point> > &v_vSamples) const {
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

void svmModel::computeKernelValuesOnFly(const vector<vector<float_point> > &samples,
                                        const vector<vector<float_point> > &supportVectors,
                                        vector<float_point> &kernelValues) const {
    kernelValues.resize(samples.size()*supportVectors.size());
    for (int i = 0; i < samples.size(); ++i) {
        for (int j = 0; j < supportVectors.size(); ++j) {
            //rbf kernel
            float_point sum = 0;
            for (int k = 0; k < supportVectors[j].size(); ++k) {
                float_point d = samples[i][k] - supportVectors[j][k];
                sum += d * d;
            }
            kernelValues[i * supportVectors.size() + j] = (float_point) exp(-param.gamma * sum);
        }
    }
}

void svmModel::predictLabels(const vector<float_point> &kernelValues, vector<float_point> &classificationResult,
                             int k) const {
    //get infomation from SVM model
    int nNumofSVs = (int) supportVectors[k].size();
    int numOfSamples = (int) (kernelValues.size() / nNumofSVs);
    classificationResult.resize(kernelValues.size());
//	int nNumofSVs = GetNumSV(pModel);
    float_point fBias = rho[k];
//	float_point fBias = *(pModel->rho);
    const float_point *pfSVsYiAlpha = coef[k].data();
//	float_point *pfSVsYiAlpha = pyfSVsYiAlpha[0];
//	int *pnSVsLabel = pModel->label;
//    float_point *pfYiAlphaofSVs;

    /*compute y_i*alpha_i*K(i, z) by GPU, where i is id of support vector.
     * pfDevSVYiAlphaHessian stores in the order of T1 sv1 sv2 ... T2 sv1 sv2 ... T3 sv1 sv2 ...
     */
    float_point *pfDevSVYiAlphaHessian;
    float_point *pfDevSVsYiAlpha;
//	int *pnDevSVsLabel;

    //if the memory is not enough for the storage when classifying all testing samples at once, divide it into multiple parts

    StorageManager *manager = StorageManager::getManager();
    long long int nMaxNumofFloatPoint = manager->GetFreeGPUMem();
    long long int nNumofPart = Ceil(nNumofSVs * numOfSamples, nMaxNumofFloatPoint);

//	cout << "cache size is: " << nMaxNumofFloatPoint << " v.s.. " << nNumofSVs * nNumofTestSamples << endl;
//	cout << "perform classification in " << nNumofPart << " time(s)" << endl;

    //allocate memory for storing classification result
    //initialise the size of each part
    int *pSizeofPart = new int[nNumofPart];
    int nAverageSize = (int) (numOfSamples / nNumofPart);
    for (int i = 0; i < nNumofPart; i++) {
        if (i != nNumofPart - 1) {
            pSizeofPart[i] = nAverageSize;
        } else {
            pSizeofPart[i] = numOfSamples - nAverageSize * i;
        }
    }

    //perform classification for each part
    for (int i = 0; i < nNumofPart; i++) {
        checkCudaErrors(cudaMalloc((void **) &pfDevSVYiAlphaHessian, sizeof(float_point) * nNumofSVs * pSizeofPart[i]));
        checkCudaErrors(cudaMalloc((void **) &pfDevSVsYiAlpha, sizeof(float_point) * nNumofSVs));
//	checkCudaErrors(cudaMalloc((void**)&pnDevSVsLabel, sizeof(int) * nNumofSVs));

        checkCudaErrors(cudaMemset(pfDevSVYiAlphaHessian, 0, sizeof(float_point) * nNumofSVs * pSizeofPart[i]));
        checkCudaErrors(cudaMemset(pfDevSVsYiAlpha, 0, sizeof(float_point) * nNumofSVs));
//	checkCudaErrors(cudaMemset(pnDevSVsLabel, 0, sizeof(int) * nNumofSVs));

        checkCudaErrors(cudaMemcpy(pfDevSVYiAlphaHessian, kernelValues.data() + i * nAverageSize * nNumofSVs,
                                   sizeof(float_point) * nNumofSVs * pSizeofPart[i], cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy(pfDevSVsYiAlpha, pfSVsYiAlpha, sizeof(float_point) * nNumofSVs, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(pnDevSVsLabel, pnSVsLabel, sizeof(int) * nNumofSVs, cudaMemcpyHostToDevice));

        //compute y_i*alpha_i*K(i, z)
        int nVecMatxMulGridDimY = pSizeofPart[i];
        int nVecMatxMulGridDimX = Ceil(nNumofSVs, BLOCK_SIZE);
        dim3 vecMatxMulGridDim(nVecMatxMulGridDimX, nVecMatxMulGridDimY);
        VectorMatrixMul << < vecMatxMulGridDim, BLOCK_SIZE >> >
                                                (pfDevSVsYiAlpha, pfDevSVYiAlphaHessian, pSizeofPart[i], nNumofSVs);

        //perform classification
        ComputeClassLabel(pSizeofPart[i], pfDevSVYiAlphaHessian,
                          nNumofSVs, fBias, classificationResult.data() + i * nAverageSize);


        //free memory
        checkCudaErrors(cudaFree(pfDevSVYiAlphaHessian));
        pfDevSVYiAlphaHessian = NULL;
        checkCudaErrors(cudaFree(pfDevSVsYiAlpha));
//	checkCudaErrors(cudaFree(pnDevSVsLabel));
    }
}

/*
 * @brief: compute/predict the labels of testing samples
 * @output: a set of class labels, associated to testing samples
 */
float_point *svmModel::ComputeClassLabel(int nNumofTestSamples,
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

