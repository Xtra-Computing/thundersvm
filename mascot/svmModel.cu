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
#include <zconf.h>
#include <cuda_profiler_api.h>
#include "trainingFunction.h"
#include "multiSmoSolver.h"
#include<map>

//#include "sigmoidTrainGPUHelper.h"

//todo move these kernel functions to a proper file
__global__ void rbfKernel(const float_point *sampleSelfDot, int numOfSamples,
                          const float_point *svMapSelfDot, int svMapSize,
                          float_point *kernelValues, float_point gamma) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / svMapSize;
    int SVId = idx % svMapSize;
    if (sampleId < numOfSamples) {
        float_point sampleDot = sampleSelfDot[sampleId];
        float_point svDot = svMapSelfDot[SVId];
        float_point dot = kernelValues[idx];
        kernelValues[idx] = expf(-gamma * (sampleDot + svDot - 2 * dot));
    }
};

__global__ void sumKernelValues(const float_point *kernelValues, int numOfSamples, int svMapSize, int cnr2,
                                const int *svIndex, const float_point *coef,
                                const int *start, const int *count,
                                const float_point *bias, float_point *decValues) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int sampleId = idx / cnr2;
    int modelId = idx % cnr2;
    if (sampleId < numOfSamples) {
        float_point sum = 0;
        const float_point *kernelValue = kernelValues + sampleId * svMapSize;
        int si = start[modelId];
        int ci = count[modelId];
        for (int i = 0; i < ci; ++i) {
            sum += coef[si + i] * kernelValue[svIndex[si + i]];
        }
        decValues[idx] = sum - bias[modelId];
    }
}

SvmModel::~SvmModel() {
    checkCudaErrors(cudaFree(devCoef));
    checkCudaErrors(cudaFree(devStart));
    checkCudaErrors(cudaFree(devCount));
    checkCudaErrors(cudaFree(devProbA));
    checkCudaErrors(cudaFree(devProbB));
    checkCudaErrors(cudaFree(devRho));
    checkCudaErrors(cudaFree(devSVMapVal));
    checkCudaErrors(cudaFree(devSVMapValSelfDot));
    checkCudaErrors(cudaFree(devSVMapRowPtr));
    checkCudaErrors(cudaFree(devSVMapColInd));
    checkCudaErrors(cudaFree(devSVIndex));
    if (svMapCSRMat) delete svMapCSRMat;
}

unsigned int SvmModel::getK(int i, int j) const {
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void SvmModel::fit(const SvmProblem &problem, const SVMParam &param) {
    //reset model to fit a new SvmProblem
    nrClass = problem.getNumOfClasses();
    cnr2 = (nrClass) * (nrClass - 1) / 2;
    numOfSVs = 0;
    numOfFeatures = 0;
    coef.clear();
    rho.clear();
    probA.clear();
    probB.clear();
    svIndex.clear();
    svMap.clear();
    label.clear();
    start.clear();
    count.clear();
    probability = false;

    coef.resize(cnr2);
    rho.resize(cnr2);
    probA.resize(cnr2);
    probB.resize(cnr2);
    svIndex.resize(cnr2);

    this->param = param;
    label = problem.label;
    numOfFeatures = problem.getNumOfFeatures();

    MultiSmoSolver multiSmoSolver(problem,*this,param);
    multiSmoSolver.solve();
    int _start = 0;
    for (int i = 0; i < cnr2; ++i) {
        start.push_back(_start);
        count.push_back(svIndex[i].size());
        _start += count[i];
    }
    transferToDevice();
}

void SvmModel::transferToDevice() {
    //convert svMap to csr matrix then copy it to device
    svMapCSRMat = new CSRMatrix(svMap,numOfFeatures);
    int nnz = svMapCSRMat->getNnz();
    checkCudaErrors(cudaMalloc((void **) &devSVMapVal, sizeof(float_point) * nnz));
    checkCudaErrors(cudaMalloc((void **) &devSVMapValSelfDot, sizeof(float_point) * svMapCSRMat->getNumOfSamples()));
    checkCudaErrors(cudaMalloc((void **) &devSVMapRowPtr, sizeof(int) * (svMapCSRMat->getNumOfSamples() + 1)));
    checkCudaErrors(cudaMalloc((void **) &devSVMapColInd, sizeof(int) * nnz));
    checkCudaErrors(
            cudaMemcpy(devSVMapVal, svMapCSRMat->getCSRVal(), sizeof(float_point) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(devSVMapValSelfDot, svMapCSRMat->getCSRValSelfDot(),
                       sizeof(float_point) * svMapCSRMat->getNumOfSamples(), cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(devSVMapRowPtr, svMapCSRMat->getCSRRowPtr(), sizeof(int) * (svMapCSRMat->getNumOfSamples() + 1),
                       cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devSVMapColInd, svMapCSRMat->getCSRColInd(), sizeof(int) * nnz, cudaMemcpyHostToDevice));

    //flat svIndex then copy in to device
    checkCudaErrors(cudaMalloc((void **) &devSVIndex, sizeof(int) * numOfSVs));
    for (int i = 0; i < cnr2; ++i) {
        checkCudaErrors(cudaMemcpy(devSVIndex + start[i], svIndex[i].data(), sizeof(int) * svIndex[i].size(),
                                   cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMalloc((void **) &devCoef, sizeof(float_point) * numOfSVs));
    checkCudaErrors(cudaMalloc((void **) &devStart, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devCount, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devProbA, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devProbB, sizeof(float_point) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devRho, sizeof(float_point) * cnr2));
    for (int i = 0; i < cnr2; ++i) {
        checkCudaErrors(cudaMemcpy(devCoef + start[i], coef[i].data(), sizeof(float_point) * count[i],
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(devProbA, probA.data(), sizeof(float_point) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devProbB, probB.data(), sizeof(float_point) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devStart, start.data(), sizeof(int) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devCount, count.data(), sizeof(int) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devRho, rho.data(), sizeof(float_point) * cnr2, cudaMemcpyHostToDevice));
}

//void SvmModel::gpu_sigmoid_train(
//	int l, const float_point *dec_values, const float_point *labels,
//	float_point& A, float_point& B)
//{
//
//	float_point prior1, prior0 ;
//	int max_iter=100;	// Maximal number of iterations
//	float_point min_step=1e-10;	// Minimal step taken in line search
//	float_point sigma=1e-12;	// For numerically strict PD of Hessian
//	float_point eps=1e-5;
//	float_point hiTarget=(prior1+1.0)/(prior1+2.0);
//	float_point loTarget=1/(prior0+2.0);
//	float_point fApB,g1,g2,gd,stepsize;
//	float_point newA,newB,newf;
//	int iter;
//	float_point fval = 0.0;
//	// Initial Point and Initial Fun Value
//	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
//
//	int blocknum=(l+THREAD_NUM-1)/THREAD_NUM;
//
//	cudaStream_t stream[2];
//    for(int i = 0;i < 2;i ++)
//        cudaStreamCreate(&stream[i]);
//
//	float_point *dev_prior1,*dev_prior0;
//	float_point *dev_labels,*dev_t,*dev_dec_values;
//	float_point *dev_fApB,*dev_fval,*dev_sum,*dev_d1,*dev_d2,*dev_g1,*dev_h11,*dev_h21,*dev_p,*dev_q;
//	float_point *dev_det,*dev_dA,*dev_dB,*dev_gd,*dev_newf;
//	float_point *dev_newA,*dev_newB;
//
//	checkCudaErrors(cudaMalloc((void**)&dev_sum,sizeof(float_point)*blocknum));
//	checkCudaErrors(cudaMalloc((void**)&dev_newA,sizeof(float_point)));
//	checkCudaErrors(cudaMalloc((void**)&dev_newB,sizeof(float_point)));
//	checkCudaErrors(cudaMalloc((void**)&dev_fApB,sizeof(float_point)*l));
//	checkCudaErrors(cudaMalloc((void**)&dev_fval,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_labels,sizeof(float_point)*l));
//	checkCudaErrors(cudaMalloc((void**)&dev_t,sizeof(float_point)*l));
//	checkCudaErrors(cudaMalloc((void**)&dev_dec_values,sizeof(float_point)*l));
//	checkCudaErrors(cudaMalloc((void**)&dev_p,sizeof(float_point)*l));
//	checkCudaErrors(cudaMalloc((void**)&dev_q,sizeof(float_point)*l));
//	checkCudaErrors(cudaMalloc((void**)&dev_d1,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_d2,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_g1,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_h11,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_h21,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_det,sizeof(float_point)));
//	checkCudaErrors(cudaMalloc((void**)&dev_dA,sizeof(float_point)));
//	checkCudaErrors(cudaMalloc((void**)&dev_dB,sizeof(float_point)));
//	checkCudaErrors(cudaMalloc((void**)&dev_gd,sizeof(float_point)));
//	checkCudaErrors(cudaMalloc((void**)&dev_newf,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_prior1,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMalloc((void**)&dev_prior0,sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//
//	checkCudaErrors(cudaMemcpy(dev_labels,labels,sizeof(float_point)*l,cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(dev_dec_values,dec_values,sizeof(float_point)*l,cudaMemcpyHostToDevice));
//
//	checkCudaErrors(cudaMemset(dev_fval, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMemset(dev_h11, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMemset(dev_h21, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMemset(dev_d1, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMemset(dev_d2, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMemset(dev_g1, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//	checkCudaErrors(cudaMemset(dev_newf, 0, sizeof(float_point)*(blocknum+1)*THREAD_NUM));
//
//	dev_getprior<<<blocknum+1,THREAD_NUM>>>(dev_labels,l,dev_prior1,dev_prior0);
//	dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_prior1,dev_sum,l);
//	dev_get_sum<<<1,1>>>(dev_sum,dev_prior1,blocknum);
//	dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_prior0,dev_sum,l);
//	dev_get_sum<<<1,1>>>(dev_sum,dev_prior0,blocknum);
//
//	checkCudaErrors(cudaMemcpy(&prior1,dev_prior1,sizeof(float_point),cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaMemcpy(&prior0,dev_prior0,sizeof(float_point),cudaMemcpyDeviceToHost));
//
//	//get fApB,fval
//	dev_getfApB_fval<<<blocknum,THREAD_NUM>>>(dev_fval,dev_labels,dev_t,dev_dec_values,dev_fApB,A,B,hiTarget,loTarget,l);
//	dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_fval,dev_sum,l);
//	dev_get_sum<<<1,1>>>(dev_sum,dev_fval,blocknum);//dev_get_fval_sum<<<1,1>>>(dev_fval);
//
//	checkCudaErrors(cudaFree(dev_labels));
//	for (iter=0;iter<max_iter;iter++)
//	{
//
//		if(iter>0)
//			//update newA,newB
//			dev_getfApB<<<blocknum,THREAD_NUM>>>(l,dev_fApB,dev_dec_values,A,B);
//		//get p q
//		dev_getpq<<<blocknum,THREAD_NUM>>>(l,dev_t,dev_fApB,dev_p,dev_q,dev_d1,dev_d2,dev_h11,dev_h21,dev_g1,dev_dec_values);
//		//get h11
//		dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_h11,dev_sum,l);
//		dev_get_sum<<<1,1>>>(dev_sum,dev_h11,blocknum);
//		//get h21
//		dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_h21,dev_sum,l);
//		dev_get_sum<<<1,1>>>(dev_sum,dev_h21,blocknum);
//		//get d2\h22
//		dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_d2,dev_sum,l);
//		dev_get_sum<<<1,1>>>(dev_sum,dev_d2,blocknum);//d2[0]=h22
//		//get g1
//		dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_g1,dev_sum,l);
//		dev_get_sum<<<1,1>>>(dev_sum,dev_g1,blocknum);
//		//get d1\g2
//		dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_d1,dev_sum,l);
//		dev_get_sum<<<1,1>>>(dev_sum,dev_d1,blocknum);//d1[0]=g2
//
//		checkCudaErrors(cudaMemcpy(&g1,dev_g1,sizeof(float_point),cudaMemcpyDeviceToHost));
//		checkCudaErrors(cudaMemcpy(&g2,dev_d1,sizeof(float_point),cudaMemcpyDeviceToHost));
//		// Stopping Criteria
//		if (fabs(g1)<eps && fabs(g2)<eps)
//			break;
//
//
//		// Finding Newton direction: -inv(H') * g
//		dev_get_det<<<1,1>>>(sigma,dev_h11,dev_d2,dev_h21,dev_det);
//		//?????????????
//	    dev_getdA<<<1,1,0,stream[0]>>>(dev_dA,dev_det,dev_d2,dev_h21,dev_g1,dev_d1);
//		dev_getdB<<<1,1,0,stream[1]>>>(dev_dB,dev_det,dev_h11,dev_h21,dev_g1,dev_d1);
//		dev_getgd<<<1,1>>>(dev_gd,dev_dA,dev_dB,dev_g1,dev_d1);
//
//		stepsize = 1;		// Line Search
//
//		while (stepsize >= min_step)
//		{
//			//update newA newB
//			dev_updateAB<<<1,2>>>(dev_newA,dev_newB,A,B,stepsize,dev_dA,dev_dB);
//
//			// New function value
//			dev_getnewfApB<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(l,dev_fApB,dev_dec_values,dev_newA,dev_newB);
//			dev_getnewf<<<blocknum,THREAD_NUM>>>(l,dev_fApB,dev_t,dev_newf);
//			dev_paral_red_sum<<<blocknum,THREAD_NUM,THREAD_NUM*sizeof(float_point)>>>(dev_newf,dev_sum,l);//more block?
//			dev_get_sum<<<1,1>>>(dev_sum,dev_newf,blocknum);
//
//			// Check sufficient decrease
//			checkCudaErrors(cudaMemcpy(&newf,dev_newf,sizeof(float_point),cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaMemcpy(&fval,dev_fval,sizeof(float_point),cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaMemcpy(&gd,dev_gd,sizeof(float_point),cudaMemcpyDeviceToHost));
//			if (newf<fval+0.0001*(float_point)stepsize*gd)
//			{
//				cudaMemcpy(&A,dev_newA,sizeof(float_point),cudaMemcpyDeviceToHost);
//				cudaMemcpy(&B,dev_newB,sizeof(float_point),cudaMemcpyDeviceToHost);
//				fval=newf;
//				break;
//			}
//			else
//				stepsize = stepsize / 2.0;
//		}
//
//		if (stepsize < min_step)
//		{
//			info("Line search fails in two-class probability estimates\n");
//			break;
//		}
//	}
//
//	if (iter>=max_iter)
//		info("Reaching maximal iterations in two-class probability estimates\n");
//
//	checkCudaErrors(cudaFree(dev_newA));
//	checkCudaErrors(cudaFree(dev_newB));
//	checkCudaErrors(cudaFree(dev_fApB));
//	checkCudaErrors(cudaFree(dev_fval));
//	checkCudaErrors(cudaFree(dev_dec_values));
//	checkCudaErrors(cudaFree(dev_det));
//	checkCudaErrors(cudaFree(dev_dA));
//	checkCudaErrors(cudaFree(dev_dB));
//	checkCudaErrors(cudaFree(dev_gd));
//	checkCudaErrors(cudaFree(dev_newf));
//	checkCudaErrors(cudaFree(dev_t));
//	checkCudaErrors(cudaFree(dev_d1));
//	checkCudaErrors(cudaFree(dev_d2));
//	checkCudaErrors(cudaFree(dev_g1));
//	checkCudaErrors(cudaFree(dev_h11));
//	checkCudaErrors(cudaFree(dev_h21));
//	checkCudaErrors(cudaFree(dev_p));
//	checkCudaErrors(cudaFree(dev_q));
//	checkCudaErrors(cudaFree(dev_sum));
//	checkCudaErrors(cudaFree(dev_prior1));
//	checkCudaErrors(cudaFree(dev_prior0));
//}


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

void SvmModel::addBinaryModel(const SvmProblem &problem, const vector<int> &svIndex, const vector<float_point> &coef,
                              float_point rho, int i,
                              int j) {
    static map<int, int> indexMap;
    int k = getK(i, j);
    this->coef[k] = coef;
    for (int l = 0; l < svIndex.size(); ++l) {
        int originalIndex = problem.originalIndex[svIndex[l]];
        if (indexMap.find(originalIndex) != indexMap.end()) {
        } else {
            indexMap[originalIndex] = svMap.size();
            svMap.push_back(problem.v_vSamples[svIndex[l]]);
        }
        this->svIndex[k].push_back(indexMap[originalIndex]);
    }
    this->rho[k] = rho;
    numOfSVs += svIndex.size();
}

void
SvmModel::predictValues(const vector<vector<svm_node> > &v_vSamples,
                        vector<vector<float_point> > &decisionValues) const {
    //copy samples to device
    CSRMatrix sampleCSRMat(v_vSamples, numOfFeatures);
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
    checkCudaErrors(
            cudaMemcpy(devSampleRowPtr, sampleCSRMat.getCSRRowPtr(), sizeof(int) * (sampleCSRMat.getNumOfSamples() + 1),
                       cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(devSampleColInd, sampleCSRMat.getCSRColInd(), sizeof(int) * sampleNnz, cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    float_point *devKernelValues;
    checkCudaErrors(cudaMalloc((void **) &devKernelValues,
                               sizeof(float_point) * v_vSamples.size() * svMap.size()));

    //dot product between sv and sample
    CSRMatrix::CSRmm2Dense(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                           sampleCSRMat.getNumOfSamples(), svMapCSRMat->getNumOfSamples(),
                           svMapCSRMat->getNumOfFeatures(),
                           descr, sampleNnz, devSampleVal, devSampleRowPtr, devSampleColInd,
                           descr, svMapCSRMat->getNnz(), devSVMapVal, devSVMapRowPtr, devSVMapColInd,
                           devKernelValues);

    //obtain exp(-gamma*(a^2+b^2-2ab))
    int numOfBlock = Ceil(v_vSamples.size() * svMap.size(), BLOCK_SIZE);
    rbfKernel << < numOfBlock, BLOCK_SIZE >> >
                               (devSampleValSelfDot, sampleCSRMat.getNumOfSamples(),
                                       devSVMapValSelfDot, svMapCSRMat->getNumOfSamples(),
                                       devKernelValues, param.gamma);

    //sum kernel values of each model then obtain decision values
    numOfBlock = Ceil(v_vSamples.size() * cnr2, BLOCK_SIZE);
    float_point *devDecisionValues;
    checkCudaErrors(cudaMalloc((void **) &devDecisionValues, sizeof(float_point) * v_vSamples.size() * cnr2));
    sumKernelValues << < numOfBlock, BLOCK_SIZE >> > (devKernelValues, v_vSamples.size(),
            svMapCSRMat->getNumOfSamples(), cnr2, devSVIndex, devCoef, devStart, devCount, devRho, devDecisionValues);
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

vector<int> SvmModel::predict(const vector<vector<svm_node> > &v_vSamples, bool probability) const {
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

vector<vector<float_point> > SvmModel::predictProbability(const vector<vector<svm_node> > &v_vSamples) const {
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


