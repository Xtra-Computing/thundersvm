/*
 * powerOfTwo.h
 *
 *  Created on: 2 Aug 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef POWEROFTWO_H_
#define POWEROFTWO_H_

#include <cuda.h>
#include "powerOfTwo.h"

inline int floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127);
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}


inline bool isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

/**
 * @brief: useful for configuring small kernels for reduction/sum/prefix-sum
 */
inline void smallReductionKernelConf(unsigned int &numThreads, int numElem)
{
	if (isPowerOfTwo(numElem))
		numThreads = numElem / 2;//only one block and
	else
		numThreads = floorPow2(numElem);//a few threads only have one element to process.
}


#endif /* POWEROFTWO_H_ */
