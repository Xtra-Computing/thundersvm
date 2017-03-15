#ifndef GETCUDAERROR_H_
#define GETCUDAERROR_H_

#define _DEBUG
#ifndef _DEBUG
	#define GETERROR(x)		((void)0)
#else
	#define GETERROR(x)\
    		if(cudaGetLastError() != cudaSuccess){printf("cuda error in: %s\n", x); exit(0);}\
			NULL;
#endif//_DEBUG

#endif /*GETCUDAERROR_H_*/
