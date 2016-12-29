/*
 * initCuda.h
 *
 *  Created on: 10/12/2014
 *      Author: Zeyi Wen
 */

#ifndef INITCUDA_H_
#define INITCUDA_H_

bool InitCUDA(CUcontext &context, char gpuType = 'T');
bool ReleaseCuda(CUcontext &context);

#endif /* INITCUDA_H_ */
