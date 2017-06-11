/**
 * constant.h
 * Created on: May 22, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef CONSTANT_H_
#define CONSTANT_H_

#include <math.h>
#include "host_constant.h"

#define BATCH_WRITE_SIZE 256000000//4096000000	//4GB for each batch write
#define MAX_SIZE_PER_READ 3072000000 //3GB for each batch read
#define NORMAL_NUMOF_DIM 1204000
#define MAX_SHARED_MEM 49152		//shared memory per block for GTX 480 and Tesla C2075 is 48KB

#define NOREDUCE 0x00000000
#define REDUCE0  0x00000001
#define REDUCE1  0x00000002

#define CACHE_SIZE 1024

//constant in device
#define UNROLL_REDUCE 7
#define TASK_OF_THREAD 8
#define TASK_OF_THREAD_LOW 64
#define TASK_OF_THREAD_UP 64
#define MIN_BLOCK_SIZE 64			//Minimum block size
//#define BLOCK_SIZE 128//Block_Size shouldn't larger than 1024
#define NUM_OF_BLOCK 65535			//Maximum number of blocks per dimension
#define MAX_GRID_SIZE_PER_DIM 65535 //Maximum size for each dimension of grid


#endif /* CONSTANT_H_ */
