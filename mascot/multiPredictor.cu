/*
 * multiPredictor.cu
 *
 *  Created on: 1 Jan 2017
 *      Author: Zeyi Wen
 */

#include <cuda.h>
#include <helper_cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <zconf.h>
#include <cuda_profiler_api.h>
#include <assert.h>
#include "multiPredictor.h"
#include "predictionGPUHelper.h"
#include "../svm-shared/constant.h"

float_point MultiPredictor::sigmoidPredict(float_point decValue, float_point A, float_point B) const {
    double fApB = decValue * A + B;
    // 1-p used later; avoid catastrophic cancellation
    if (fApB >= 0)
        return exp(-fApB) / (1.0 + exp(-fApB));
    else
        return 1.0 / (1 + exp(fApB));
}

void MultiPredictor::multiClassProbability(const vector<vector<float_point> > &r, vector<float_point> &p) const {
	int nrClass = model.nrClass;
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

vector<vector<float_point> > MultiPredictor::predictProbability(const vector<vector<KeyValue> > &v_vSamples) const {
	int nrClass = model.nrClass;
    vector<vector<float_point> > result;
    vector<vector<float_point> > decValues;
    computeDecisionValues(v_vSamples, decValues);
    for (int l = 0; l < v_vSamples.size(); ++l) {
        vector<vector<float_point> > r(nrClass, vector<float_point>(nrClass));
        double min_prob = 1e-7;
        int k = 0;
        for (int i = 0; i < nrClass; i++)
            for (int j = i + 1; j < nrClass; j++) {
                r[i][j] = min(
                        max(sigmoidPredict(decValues[l][k], model.probA[k], model.probB[k]), min_prob), 1 - min_prob);
                r[j][i] = 1 - r[i][j];
                k++;
            }
        vector<float_point> p(nrClass);
        multiClassProbability(r, p);
        result.push_back(p);
    }
    return result;
}

/**
 * @brief: compute the decision value
 */
void MultiPredictor::computeDecisionValues(const vector<vector<KeyValue> > &v_vSamples,
                        		   vector<vector<float_point> > &decisionValues) const {
    //copy samples to device
    CSRMatrix sampleCSRMat(v_vSamples, model.numOfFeatures);
    float_point *devSampleVal;
    float_point *devSampleValSelfDot;
    int *devSampleRowPtr;
    int *devSampleColInd;
    int sampleNnz = sampleCSRMat.getNnz();
    checkCudaErrors(cudaMalloc((void **) &devSampleVal, sizeof(float_point) * sampleNnz));
    checkCudaErrors(cudaMalloc((void **) &devSampleValSelfDot, sizeof(float_point) * sampleCSRMat.getNumOfSamples()));
    checkCudaErrors(cudaMalloc((void **) &devSampleRowPtr, sizeof(int) * (sampleCSRMat.getNumOfSamples() + 1)));
    checkCudaErrors(cudaMalloc((void **) &devSampleColInd, sizeof(int) * sampleNnz));
    checkCudaErrors(cudaMemcpy(devSampleVal, sampleCSRMat.getCSRVal(), sizeof(float_point) * sampleNnz,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devSampleValSelfDot, sampleCSRMat.getCSRValSelfDot(),
                               sizeof(float_point) * sampleCSRMat.getNumOfSamples(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devSampleRowPtr, sampleCSRMat.getCSRRowPtr(),
    						   sizeof(int) * (sampleCSRMat.getNumOfSamples() + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devSampleColInd, sampleCSRMat.getCSRColInd(), sizeof(int) * sampleNnz,
    						   cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    float_point *devKernelValues;
    checkCudaErrors(cudaMalloc((void **) &devKernelValues,
    						   sizeof(float_point) * v_vSamples.size() * model.svMap.size()));

    //dot product between sv and sample
    CSRMatrix::CSRmm2Dense(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                           sampleCSRMat.getNumOfSamples(), model.svMapCSRMat->getNumOfSamples(),
						   model.svMapCSRMat->getNumOfFeatures(),
                           descr, sampleNnz, devSampleVal, devSampleRowPtr, devSampleColInd,
                           descr, model.svMapCSRMat->getNnz(), model.devSVMapVal, model.devSVMapRowPtr, model.devSVMapColInd,
                           devKernelValues);

    //obtain exp(-gamma*(a^2+b^2-2ab))
    int numOfBlock = Ceil(v_vSamples.size() * model.svMap.size(), BLOCK_SIZE);
    rbfKernel<<<numOfBlock, BLOCK_SIZE>>>(devSampleValSelfDot, sampleCSRMat.getNumOfSamples(),
                            		      model.devSVMapValSelfDot, model.svMapCSRMat->getNumOfSamples(),
										  devKernelValues, model.param.gamma);

    //sum kernel values of each model then obtain decision values
    int cnr2 = model.cnr2;
    numOfBlock = Ceil(v_vSamples.size() * cnr2, BLOCK_SIZE);
    float_point *devDecisionValues;
    checkCudaErrors(cudaMalloc((void **) &devDecisionValues, sizeof(float_point) * v_vSamples.size() * cnr2));
    sumKernelValues<<<numOfBlock, BLOCK_SIZE>>>(devKernelValues, v_vSamples.size(),
    				model.svMapCSRMat->getNumOfSamples(), cnr2, model.devSVIndex,
					model.devCoef, model.devStart, model.devCount, model.devRho, devDecisionValues);
    float_point *tempDecValues = new float_point[v_vSamples.size() * cnr2];
    checkCudaErrors(cudaMemcpy(tempDecValues, devDecisionValues,
                               sizeof(float_point) * v_vSamples.size() * cnr2, cudaMemcpyDeviceToHost));
    decisionValues = vector<vector<float_point> >(v_vSamples.size(), vector<float_point>(cnr2));
    for (int i = 0; i < decisionValues.size(); ++i) {
        memcpy(decisionValues[i].data(), tempDecValues + i * cnr2, sizeof(float_point) * cnr2);
    }
    delete[] tempDecValues;
    checkCudaErrors(cudaFree(devDecisionValues));
    checkCudaErrors(cudaFree(devKernelValues));
    checkCudaErrors(cudaFree(devSampleVal));
    checkCudaErrors(cudaFree(devSampleValSelfDot));
    checkCudaErrors(cudaFree(devSampleRowPtr));
    checkCudaErrors(cudaFree(devSampleColInd));
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

/**
 * @brief: predict the label of the instances
 * @param: vnOriginalLabel is for computing errors of sub-classifier.
 */
vector<int> MultiPredictor::predict(const vector<vector<KeyValue> > &v_vSamples, const vector<int> &vnOriginalLabel) const{
	int nrClass = model.nrClass;
	bool probability = model.isProbability();
    vector<int> labels;
    if (!probability) {
        vector<vector<float_point> > decisionValues;
        computeDecisionValues(v_vSamples, decisionValues);
        for (int l = 0; l < v_vSamples.size(); ++l) {
        	if(!vnOriginalLabel.empty())//want to measure sub-classifier error
        	{
        		//update model labeling information
                int rawLabel = vnOriginalLabel[l];
				int originalLabel = -1;
                for(int pos = 0; pos < model.label.size(); pos++){
                    if(model.label[pos] == rawLabel)
                        originalLabel = pos;
                }
				model.missLabellingMatrix[originalLabel][originalLabel]++;//increase the total occurrence of a label.
				int k = 0;
	            for (int i = 0; i < nrClass; ++i) {
	                for (int j = i + 1; j < nrClass; ++j) {
	                	int labelViaBinary = j;
	                    if (decisionValues[l][k++] > 0)
	                    	labelViaBinary = i;

	    				if(i == originalLabel || j == originalLabel){
	    					if(labelViaBinary != originalLabel)//miss classification
	    						model.missLabellingMatrix[originalLabel][labelViaBinary]++;
	    				}
	                }
	            }
        	}

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
            labels.push_back(model.label[maxVoteClass]);
        }
    } else {
        printf("predict with probability\n");
        assert(model.probability);
        vector<vector<float_point> > prob = predictProbability(v_vSamples);
        // todo select max using GPU
        for (int i = 0; i < v_vSamples.size(); ++i) {
            int maxProbClass = 0;
            for (int j = 0; j < nrClass; ++j) {
                if (prob[i][j] > prob[i][maxProbClass])
                    maxProbClass = j;
            }
            labels.push_back(model.label[maxProbClass]);
        }
    }
    return labels;
}
