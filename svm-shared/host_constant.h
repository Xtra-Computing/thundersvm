/*
 * host_constant.h
 *
 *  Created on: 31/10/2015
 *      Author: Zeyi Wen
 */

#ifndef HOST_CONSTANT_H_
#define HOST_CONSTANT_H_

//for hessian matrix
#define HESSIAN_FILE 		"bin/hessian.bin"
#define HESSIAN_DIAG_FILE 	"bin/hessian_diag.bin"

#define RBFKERNEL		"RBF"
#define SVMLINEAR 		"Linear"
#define SVMPOLYNOMIAL	"Polynomial"
#define SVMSIGMOID		"Sigmoid"

#define RAM_SIZE 5

#define OUTPUT_FILE	"result.txt"

#define TAU 0.001//1e-5//1e-12
#define EPS 0.001//0.0001
#define ITERATION_FACTOR 50	//maximum iteration

#endif /* HOST_CONSTANT_H_ */
