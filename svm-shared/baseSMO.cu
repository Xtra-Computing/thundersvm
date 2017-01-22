/*
 * baseSMO.cu
 *  @brief: definition of some sharable functions of smo solver
 *  Created on: 24 Dec 2016
 *      Author: Zeyi Wen
 */

#include "baseSMO.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include "smoGPUHelper.h"
#include "../SharedUtility/Timer.h"

/**
 * @brief: initialise some variables of smo solver
 */
void BaseSMO::InitSolver(int nNumofTrainingIns)
{
	alpha = vector<float_point>(nNumofTrainingIns, 0);

	//configure cuda kernel
	numOfBlock = Ceil(nNumofTrainingIns, BLOCK_SIZE);
	gridSize = dim3(numOfBlock > NUM_OF_BLOCK ? NUM_OF_BLOCK : numOfBlock, Ceil(numOfBlock, NUM_OF_BLOCK));
	//allocate device memory for min/max search
	checkCudaErrors(cudaMalloc((void**)&devBlockMin, sizeof(float_point) * numOfBlock));
	checkCudaErrors(cudaMalloc((void**)&devBlockMinGlobalKey, sizeof(int) * numOfBlock));
	//for getting maximum low G value
	checkCudaErrors(cudaMalloc((void**)&devBlockMinYiGValue, sizeof(float_point) * numOfBlock));
	checkCudaErrors(cudaMalloc((void**)&devMinValue, sizeof(float_point)));
	checkCudaErrors(cudaMalloc((void**)&devMinKey, sizeof(int)));

	checkCudaErrors(cudaMallocHost((void **) &hostBuffer, sizeof(float_point) * 5));
	checkCudaErrors(cudaMalloc((void**)&devBuffer, sizeof(float_point) * 5));//only need 4 float_points

	//diagonal is frequently used in training.
	hessianDiag = new float_point[nNumofTrainingIns];
    checkCudaErrors(cudaMalloc((void **) &devHessianDiag, sizeof(float_point) * nNumofTrainingIns));
}

/**
 * @brief: release solver memory
 */
void BaseSMO::DeInitSolver()
{
    checkCudaErrors(cudaFree(devBlockMin));
    checkCudaErrors(cudaFree(devBlockMinGlobalKey));
    checkCudaErrors(cudaFree(devBlockMinYiGValue));
    checkCudaErrors(cudaFree(devMinValue));
    checkCudaErrors(cudaFree(devMinKey));
    checkCudaErrors(cudaFree(devBuffer));
    checkCudaErrors(cudaFreeHost(hostBuffer));
    checkCudaErrors(cudaFree(devHessianDiag));
    delete[] hessianDiag;
}

/**
 * @brief: select the first instance in SMO
 */
void BaseSMO::SelectFirst(int numTrainingInstance, float_point CforPositive)
{
    TIMER_START(selectTimer)
	GetBlockMinYiGValue<<<gridSize, BLOCK_SIZE>>>(devYiGValue, devAlpha, devLabel, CforPositive,
														   numTrainingInstance, devBlockMin, devBlockMinGlobalKey);
	//global reducer
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMin, devBlockMinGlobalKey, numOfBlock, devYiGValue, NULL, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 2, cudaMemcpyDeviceToHost);
	IdofInstanceOne = (int)hostBuffer[0];
    TIMER_STOP(selectTimer)

	devHessianInstanceRow1 = ObtainRow(numTrainingInstance);
}

/**
 * @breif: select the second instance in SMO
 */
void BaseSMO::SelectSecond(int numTrainingInstance, float_point CforNegative)
{
    TIMER_START(selectTimer)
	float_point fUpSelfKernelValue = 0;
	fUpSelfKernelValue = hessianDiag[IdofInstanceOne];

	//for selecting the second instance
	float_point fMinValue;
	fMinValue = hostBuffer[1];
	upValue = -fMinValue;

	//get block level min (-b_ij*b_ij/a_ij)
	GetBlockMinLowValue<<<gridSize, BLOCK_SIZE>>>
						   (devYiGValue, devAlpha, devLabel, CforNegative, numTrainingInstance, devHessianDiag,
							devHessianInstanceRow1, upValue, fUpSelfKernelValue, devBlockMin, devBlockMinGlobalKey,
							devBlockMinYiGValue);

	//get global min
	GetGlobalMin<<<1, BLOCK_SIZE>>>
					(devBlockMin, devBlockMinGlobalKey,
					 numOfBlock, devYiGValue, devHessianInstanceRow1, devBuffer);

	//get global min YiFValue
	//0 is the size of dynamically allocated shared memory inside kernel
	GetGlobalMin<<<1, BLOCK_SIZE>>>(devBlockMinYiGValue, numOfBlock, devBuffer);

	//copy result back to host
	cudaMemcpy(hostBuffer, devBuffer, sizeof(float_point) * 4, cudaMemcpyDeviceToHost);
    TIMER_STOP(selectTimer)
}

/**
 * @brief: update two weights
 */
void BaseSMO::UpdateTwoWeight(float_point fMinLowValue, float_point fMinValue, int nHessianRowOneInMatrix,
                                     int nHessianRowTwoInMatrix, float_point fKernelValue, float_point &fY1AlphaDiff,
                                     float_point &fY2AlphaDiff, const int *label, float_point C) {
    //get YiGValue for sample one and two
    float_point fAlpha2 = 0;
    float_point fYiFValue2 = 0;
    fAlpha2 = alpha[IdofInstanceTwo];	//reserved for svm regression
    fYiFValue2 = fMinLowValue;

    //get alpha values of sample
    float_point fAlpha1 = 0;
    float_point fYiFValue1 = 0;
    fAlpha1 = alpha[IdofInstanceOne];	//reserved for svm regression
    fYiFValue1 = fMinValue;

    //Get K(x_up, x_up), and K(x_low, x_low)
    float_point fDiag1 = 0, fDiag2 = 0;
    fDiag1 = hessianDiag[nHessianRowOneInMatrix];
    fDiag2 = hessianDiag[nHessianRowTwoInMatrix];

    //get labels of sample one and two
    int nLabel1 = 0, nLabel2 = 0;
    nLabel1 = label[IdofInstanceOne];
    nLabel2 = label[IdofInstanceTwo];

    //compute eta
    float_point eta = fDiag1 + fDiag2 - 2 * fKernelValue;
    if (eta <= 0)
        eta = TAU;

    float_point fCost1, fCost2;
//	fCost1 = Get_C(nLabel1);
//	fCost2 = Get_C(nLabel2);
    fCost1 = fCost2 = C;

    //keep old yi*alphas
    fY1AlphaDiff = nLabel1 * fAlpha1;
    fY2AlphaDiff = nLabel2 * fAlpha2;

    //get new alpha values
    int nSign = nLabel2 * nLabel1;
    if (nSign < 0) {
        float_point fDelta = (-nLabel1 * fYiFValue1 - nLabel2 * fYiFValue2) / eta; //(-fYiFValue1 - fYiFValue2) / eta;
        float_point fAlphaDiff = fAlpha1 - fAlpha2;
        fAlpha1 += fDelta;
        fAlpha2 += fDelta;

        if (fAlphaDiff > 0) {
            if (fAlpha2 < 0) {
                fAlpha2 = 0;
                fAlpha1 = fAlphaDiff;
            }
        } else {
            if (fAlpha1 < 0) {
                fAlpha1 = 0;
                fAlpha2 = -fAlphaDiff;
            }
        }

        if (fAlphaDiff > fCost1 - fCost2) {
            if (fAlpha1 > fCost1) {
                fAlpha1 = fCost1;
                fAlpha2 = fCost1 - fAlphaDiff;
            }
        } else {
            if (fAlpha2 > fCost2) {
                fAlpha2 = fCost2;
                fAlpha1 = fCost2 + fAlphaDiff;
            }
        }
    } //end if nSign < 0
    else {
        float_point fDelta = (nLabel1 * fYiFValue1 - nLabel2 * fYiFValue2) / eta;
        float_point fSum = fAlpha1 + fAlpha2;
        fAlpha1 -= fDelta;
        fAlpha2 += fDelta;

        if (fSum > fCost1) {
            if (fAlpha1 > fCost1) {
                fAlpha1 = fCost1;
                fAlpha2 = fSum - fCost1;
            }
        } else {
            if (fAlpha2 < 0) {
                fAlpha2 = 0;
                fAlpha1 = fSum;
            }
        }
        if (fSum > fCost2) {
            if (fAlpha2 > fCost2) {
                fAlpha2 = fCost2;
                fAlpha1 = fSum - fCost2;
            }
        } else {
            if (fAlpha1 < 0) {
                fAlpha1 = 0;
                fAlpha2 = fSum;
            }
        }
    }//end get new alpha values

    alpha[IdofInstanceOne] = fAlpha1;
    alpha[IdofInstanceTwo] = fAlpha2;

    //get alpha difference
    fY1AlphaDiff = nLabel1 * fAlpha1 - fY1AlphaDiff; //(alpha1' - alpha1) * y1
    fY2AlphaDiff = nLabel2 * fAlpha2 - fY2AlphaDiff;
}

/*
 * @brief: update the optimality indicator
 */
void BaseSMO::UpdateYiGValue(int numTrainingInstance, float_point fY1AlphaDiff, float_point fY2AlphaDiff)
{
    float_point fAlpha1 = alpha[IdofInstanceOne];
    float_point fAlpha2 = alpha[IdofInstanceTwo];
    //update yiFvalue
    //copy new alpha values to device
    hostBuffer[0] = IdofInstanceOne;
    hostBuffer[1] = fAlpha1;
    hostBuffer[2] = IdofInstanceTwo;
    hostBuffer[3] = fAlpha2;
    checkCudaErrors(cudaMemcpy(devBuffer, hostBuffer, sizeof(float_point) * 4, cudaMemcpyHostToDevice));
    UpdateYiFValueKernel <<< gridSize, BLOCK_SIZE >>> (devAlpha, devBuffer, devYiGValue,
            devHessianInstanceRow1, devHessianInstanceRow2,
            fY1AlphaDiff, fY2AlphaDiff, numTrainingInstance);
    cudaDeviceSynchronize();
}
