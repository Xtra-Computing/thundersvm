/**
 * constant.h
 * Created on: May 22, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef CONSTANT_H_
#define CONSTANT_H_

#include <math.h>
//for experiment

//for hessian matrix
#define HESSIAN_FILE "/media/zeyi/SSD/hessian_par.bin"
#define HESSIAN_DIAG_FILE "/media/zeyi/SSD/hessian.bin"
#define BATCH_WRITE_SIZE 256000000//4096000000	//4GB for each batch write
#define MAX_SIZE_PER_READ 3072000000 //3GB for each batch read
#define NORMAL_NUMOF_DIM 1204000
#define MAX_SHARED_MEM 49152		//shared memory per block for GTX 480 and Tesla C2075 is 48KB

#define NOREDUCE 0x00000000
#define REDUCE0  0x00000001
#define REDUCE1  0x00000002

#define RBFKERNEL	"RBF"
#define LINEAR 		"Linear"
#define POLYNOMIAL	"Polynomial"
#define SIGMOID		"Sigmoid"

#define RAM_SIZE 5
#define CACHE_SIZE 200
//#define NUMOFSAMPLE 32561			//49749 web-a.dst; 31420 K9.data; 72309 real-sim;
									//32561 adult a9a.txt; 60000 mnist.scale; 7291 usps; 400,000 (30,60,90,120) epsilon_normalized
//#define TRAINING_DATA "dataset/a9a.txt"
//#define NUMOF_GAMMA	1				//1 is for web-a.dst and a9a.txt; 7 for K9.data;
//#define	START_GAMMA	0.5			//7.8125 web-a.dst; pow(2, -15) K9.data; 0.5 a9a.txt;
								//0.125 mnist.scale and usps; 1 epsilon_normalized
//#define NUMOF_COST 1				//1 is for web-a.dst and a9a.txt; 11 for K9.data;
//#define COST_POSITIVE_SAMPLE 64		//64 web-a.dst; pow(2.0, -5) K9.data, 32768=2^15 seems good for C; 100 a9a.txt; 10 mnist.scale and usps
//#define COST_NEGATIVE_SAMPLE 16		//pow(2.0, -5), epsilon 0.01
#define OUTPUT_FILE	"result.txt"

#define TAU 0.001//1e-5//1e-12
#define EPS 0.0001//0.0001
#define ITERATION_FACTOR 50	//maximum iteration


//constant in device
#define UNROLL_REDUCE 7
#define TASK_OF_THREAD 8
#define TASK_OF_THREAD_LOW 64
#define TASK_OF_THREAD_UP 64
#define MIN_BLOCK_SIZE 64			//Minimum block size
#define BLOCK_SIZE 128				//Block_Size shouldn't larger than 1024
#define NUM_OF_BLOCK 65535			//Maximum number of blocks per dimension
#define MAX_GRID_SIZE_PER_DIM 65535 //Maximum size for each dimension of grid


#endif /* CONSTANT_H_ */
