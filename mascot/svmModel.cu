/*
   * @author: created by ss on 16-11-2.
   * @brief: multi-class svm training, prediction, svm with probability output
   *
*/

#include <map>
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include "svmModel.h"
#include "svmPredictor.h"
#include "multiSmoSolver.h"
#include "multiPredictor.h"
#include "trainClassifier.h"
#include "../svm-shared/HessianIO/deviceHessian.h"
#include "../svm-shared/storageManager.h"
#include <cstring>
#include <time.h>//#include "sigmoidTrainGPUHelper.h"

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

/*
 * @brief: get the classifier id based on i and j.
 */
uint SvmModel::getK(int i, int j) const {
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void SvmModel::fit(const SvmProblem &problem, const SVMParam &param) {
    //reset model to fit a new SvmProblem
    nrClass = problem.getNumOfClasses();
    cnr2 = (nrClass) * (nrClass - 1) / 2;
    numOfSVs = 0;
    numOfFeatures = 0;
    coef.clear();
    allcoef.clear();
    rho.clear();
    probA.clear();
    probB.clear();
    svIndex.clear();
    svMap.clear();
    label.clear();
    start.clear();
    count.clear();
    probability = false;
    nSV.clear();
    nonzero.clear();

    nSV.resize(nrClass, 0);
    nonzero.resize(problem.getNumOfSamples(), false);
    coef.resize(cnr2);
    allcoef.resize(cnr2);
    rho.resize(cnr2);
    probA.resize(cnr2);
    probB.resize(cnr2);
    svIndex.resize(cnr2);

    this->param = param;
    label = problem.label;
    numOfFeatures = problem.getNumOfFeatures();

    MultiSmoSolver multiSmoSolver(problem, *this, param);
    multiSmoSolver.solve();    int _start = 0;
    for (int i = 0; i < cnr2; ++i) {
        start.push_back(_start);
        count.push_back(svIndex[i].size());
        _start += count[i];
    }
    transferToDevice();
    if (param.probability) {
        cout<<"performing probability training"<<endl;
        probability = true;
        int batch_size = 10000;
        vector<vector<real> > decValues;
        MultiPredictor predictor(*this, param);
        int begin = 0;
        while (begin < problem.getNumOfSamples()) {
            int end = min(begin + batch_size, (int) problem.getNumOfSamples());
            vector<vector<real> > decValuesPart;
            vector<vector<KeyValue> > samplesPart(problem.v_vSamples.begin() + begin, problem.v_vSamples.begin() + end);
            predictor.computeDecisionValues(samplesPart, decValuesPart);
            decValues.insert(decValues.end(), decValuesPart.begin(), decValuesPart.end());
            begin += batch_size;
        }
        int k = 0;
        for (int i = 0; i < nrClass; ++i) {
            for (int j = i+1; j < nrClass; ++j) {
                SvmProblem subProblem = problem.getSubProblem(i, j);
                vector<real> subDecValues;
                for (int l = 0; l < subProblem.getNumOfSamples(); ++l) {
                    subDecValues.push_back(decValues[subProblem.originalIndex[l]][k]);
                }
                sigmoidTrain(subDecValues.data(), subProblem.getNumOfSamples(), subProblem.v_nLabels, probA[k],
                             probB[k]);
                k++;
            }
        }
    }

}

void SvmModel::transferToDevice() {
    //convert svMap to csr matrix then copy it to device
    svMapCSRMat = new CSRMatrix(svMap, numOfFeatures);
    int nnz = svMapCSRMat->getNnz();
    checkCudaErrors(cudaMalloc((void **) &devSVMapVal, sizeof(real) * nnz));
    checkCudaErrors(cudaMalloc((void **) &devSVMapValSelfDot, sizeof(real) * svMapCSRMat->getNumOfSamples()));
    checkCudaErrors(cudaMalloc((void **) &devSVMapRowPtr, sizeof(int) * (svMapCSRMat->getNumOfSamples() + 1)));
    checkCudaErrors(cudaMalloc((void **) &devSVMapColInd, sizeof(int) * nnz));
    checkCudaErrors(
            cudaMemcpy(devSVMapVal, svMapCSRMat->getCSRVal(), sizeof(real) * nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(devSVMapValSelfDot, svMapCSRMat->getCSRValSelfDot(),
                       sizeof(real) * svMapCSRMat->getNumOfSamples(), cudaMemcpyHostToDevice));
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

    checkCudaErrors(cudaMalloc((void **) &devCoef, sizeof(real) * numOfSVs));
    checkCudaErrors(cudaMalloc((void **) &devStart, sizeof(real) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devCount, sizeof(real) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devProbA, sizeof(real) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devProbB, sizeof(real) * cnr2));
    checkCudaErrors(cudaMalloc((void **) &devRho, sizeof(real) * cnr2));
    for (int i = 0; i < cnr2; ++i) {
        checkCudaErrors(cudaMemcpy(devCoef + start[i], coef[i].data(), sizeof(real) * count[i],
                                   cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(devProbA, probA.data(), sizeof(real) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devProbB, probB.data(), sizeof(real) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devStart, start.data(), sizeof(int) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devCount, count.data(), sizeof(int) * cnr2, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devRho, rho.data(), sizeof(real) * cnr2, cudaMemcpyHostToDevice));
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


void SvmModel::sigmoidTrain(const real *decValues, const int l, const vector<int> &labels, real &A,
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

/**
  *@brief: add a binary svm model to the multi-class svm model.
**/
void SvmModel::addBinaryModel(const SvmProblem &problem, const vector<int> &svLocalIndex, const vector<real> &coef,
                              real rho, int i, int j) {
    static map<int, int> indexMap;
    int k = getK(i, j);
    this->coef[k] = coef;
    for (int l = 0; l < svLocalIndex.size(); ++l) {
        //map SV local index to the instance index (global index) in the whole training set
        int originalIndex = problem.originalIndex[svLocalIndex[l]];
        if (indexMap.find(originalIndex) != indexMap.end()) {//instance of this sv has been stored in svMap
        } else {
            indexMap[originalIndex] = svMap.size();//key is SV's global index; value is the id (in the map) for this SV instance.
            svMap.push_back(problem.v_vSamples[svLocalIndex[l]]);
        }
        this->svIndex[k].push_back(indexMap[originalIndex]);//svIndex is the id in the map.
    }
    this->rho[k] = rho;
    numOfSVs += svLocalIndex.size();
}

void SvmModel::updateAllCoef(int l, int indOffset, int nr_class, int &count, int k, const vector<int> &svIndex,
                             const vector<real> &coef, vector<int> &prob_start) {
    for (int i = 0; i < l; i++) {
        if (i + indOffset == svIndex[count]) {
            allcoef[k].push_back(coef[count++]);
            if (!nonzero[prob_start[nr_class] + i]) {
                nonzero[prob_start[nr_class] + i] = true;
                nSV[nr_class]++;
            }
        } else {
            allcoef[k].push_back(0);
        }
    }
}

void SvmModel::getModelParam(const SvmProblem &subProblem, const vector<int> &svIndex, const vector<real> &coef,
                             vector<int> &prob_start, int ci, int i, int j) {
    const unsigned int trainingSize = subProblem.getNumOfSamples();
    int k = getK(i, j);
    int count = 0;
    //update coef, nonzero, nSV, of class i
    updateAllCoef(ci, 0, i, count, k, svIndex, coef, prob_start);
    //update coef, nonzero, nSV, of class j
    updateAllCoef(trainingSize - ci, ci, j, count, k, svIndex, coef, prob_start);
}

bool SvmModel::isProbability() const {
    return probability;
}

bool SvmModel::saveLibModel(string filename, const SvmProblem &problem) {
    bool ret = false;
    ofstream libmod;
    string str = filename + ".model";
    libmod.open(str.c_str());
    if (!libmod.is_open()) {
        cout << "can't open file " << filename << endl;
        return ret;
    }
    const SVMParam &param = this->param;
    const char *sType[] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "NULL"};  /* svm_type */
    const char *kType[] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};
    libmod << "svm_type " << sType[param.svm_type] << endl;
    libmod << "kernel_type " << kType[param.kernel_type] << endl;;
    if (param.kernel_type == 1)
        libmod << "degree " << param.degree << endl;
    if (param.kernel_type == 1 || param.kernel_type == 2 || param.kernel_type == 3)/*1:poly 2:rbf 3:sigmoid*/
        libmod << "gamma " << param.gamma << endl;
    if (param.kernel_type == 1 || param.kernel_type == 3)
        libmod << "coef0 " << param.coef0 << endl;
    unsigned int nr_class = this->nrClass;
    unsigned int total_sv = this->numOfSVs;
    libmod << "nr_class " << nr_class << endl;
    libmod << "total_sv " << total_sv << endl;
    vector<real> frho = rho;
    libmod << "rho";
    for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++) {
        libmod << " " << frho[i];
    }
    libmod << endl;
    if (param.svm_type == 0) {
        libmod << "label";
        for (int i = 0; i < nr_class; i++)
            libmod << " " << label[i];
        libmod << endl;
    }
    if (this->probability) // regression has probA only
    {
        libmod << "probA";
        for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
            libmod << " " << probA[i];
        libmod << endl;
        libmod << "probB";
        for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
            libmod << " " << probB[i];
        libmod << endl;
    }
    if (param.svm_type == 0)//c-svm
    {
        libmod << "nr_sv";
        for (int i = 0; i < nr_class; i++)
            libmod << " " << nSV[i];
        libmod << endl;
    }

    libmod << "SV" << endl;
    vector<int> prob_count(problem.count);
    vector<int> prob_start(problem.start);
    for (int i = 0; i < nr_class; i++) {
        for (int k = 0; k < prob_count[i]; k++) {
            if (this->nonzero[prob_start[i] + k]) {
                for (int j = 0; j < nr_class; j++) {

                    if (i < j)
                        libmod << allcoef[this->getK(i, j)][k] << " ";

                    if (i > j)
                        libmod << allcoef[this->getK(j, i)][prob_count[j] + k] << " ";

                }

                vector<KeyValue> keyvalue(problem.v_vSamples[problem.perm[prob_start[i] + k]]);
                for (int l = 0; l < keyvalue.size(); l++) {
                    libmod << keyvalue[l].id << ":" << keyvalue[l].featureValue << " ";
                }
                libmod << endl;
            }
        }
    }
    libmod.close();
    ret = true;
    return ret;
}

void SvmModel::loadLibModel(string filename, SvmModel &model) {
    SVMParam param;
    int nr_class = 0;
    unsigned int cnr2 = 0;
    const char *sType[] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "NULL"};  /* svm_type */
    const char *kType[] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};

    ifstream ifs;
    filename = filename + ".model";
    ifs.open(filename.c_str());//"dataset/a6a.model");
    if (!ifs.is_open())
        cout << "can't open file" << endl;
    string feature;
    while (ifs >> feature) {
        // cout<<feature<<endl;
        if (feature == "svm_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (value == sType[i])
                    param.svm_type = i;
            }
        } else if (feature == "kernel_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (feature == kType[i])
                    param.kernel_type = i;
            }
        } else if (feature == "degree") {
            ifs >> param.degree;
        } else if (feature == "coef0") {
            ifs >> param.coef0;
        } else if (feature == "gamma") {
            ifs >> param.gamma;

        } else if (feature == "nr_class") {
            ifs >> model.nrClass;
            nr_class = model.nrClass;
            model.cnr2 = nr_class * (nr_class - 1) / 2;
            cnr2 = model.cnr2;
        } else if (feature == "total_sv") {
            ifs >> model.numOfSVs;
        } else if (feature == "rho") {
            vector<real> frho(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> frho[i];
            model.rho = frho;
        } else if (feature == "label") {
            vector<int> ilabel(nr_class, 0);
            for (int i = 0; i < nr_class; i++)
                ifs >> ilabel[i];
            model.label = ilabel;
        } else if (feature == "probA") {
            vector<real> fprobA(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> fprobA[i];
            model.probA = fprobA;
        } else if (feature == "probB") {
            vector<real> fprobB(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> fprobB[i];
            model.probB = fprobB;
        } else if (feature == "nr_sv") {
            vector<int> fnSV(nr_class, 0);
            for (int i = 0; i < nr_class; i++)
                ifs >> fnSV[i];
            model.nSV = fnSV;
        } else if (feature == "SV") {
            string value;
            stringstream sstr;
            real ftemp = 0;
            vector<vector<real> > v_allcoef(cnr2);
            vector<vector<KeyValue> > v_svMap(numOfSVs);
            //while(!ifs.eof()){
            int count = 0;
            //clock_t startt,endt;
            for (int k = 0; k < nr_class; k++) {
                for (int m = 0; m < model.nSV[k]; m++) {
                    for (int j = 0; j < nr_class; j++) {
                        if (k < j) {
                            ifs >> ftemp;
                            v_allcoef[getK(k, j)].push_back(ftemp);
                        }
                        if (k > j) {
                            ifs >> ftemp;
                            v_allcoef[getK(j, k)].push_back(ftemp);
                        }
                    }
                    getline(ifs, value);
                    KeyValue kv;
                    //startt=clock();
                    /*
                        int ind=value.find_first_of(" ");
                        string tempstr=value.substr(ind+1,value.size());
                        stringstream stemp;
                       while((ind=tempstr.find_first_of(" "))!=-1){
                //	cout<<"ind "<<ind<<endl;
                    string subtempstr=tempstr.substr(0,ind);
                            tempstr=tempstr.substr(ind+1,tempstr.size());
                  //          cout<<"tmpstr "<<subtempstr<<endl;
                int subind=subtempstr.find_first_of(":");
                        stemp<<subtempstr.substr(0,subind);
                            stemp>>kv.id;
                            stemp.clear();
                            stemp<<subtempstr.substr(subind+1,value.size());
                            stemp>>kv.featureValue;
                            v_svMap[count].push_back(kv);
                          // cout<<":"<<endl;
                          // cout<<subtempstr.substr(subind+1,subtempstr.size())<<endl;
                        }
                        count++;
                    */

                    sstr << value;
                    string temp;
                    stringstream stemp;
                    while (sstr >> temp) {
                        int ind = temp.find_first_of(":");
                        stemp << temp.substr(0, ind);
                        stemp >> kv.id;
                        stemp.clear();
                        stemp << temp.substr(ind + 1, value.size());
                        stemp >> kv.featureValue;
                        //cout<<kv.id<<":"<<kv.featureValue<<endl;
                        stemp.clear();
                        v_svMap[count].push_back(kv);
                    }
                    count++;
                }
            }
            //}
            //endt=clock();
            //cout<<"elapsed time "<<(endt-startt)<<endl;
            model.allcoef = v_allcoef;
            model.svMap = v_svMap;

        }//end else if

    }//end while
    model.param = param;
}
