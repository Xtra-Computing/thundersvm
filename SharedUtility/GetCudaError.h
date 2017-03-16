#ifndef GETCUDAERROR_H_
#define GETCUDAERROR_H_

#define _DEBUG
#ifndef _DEBUG
#define GETERROR(x) ((void)0)
#else
#define GETERROR(x) do {					\
		if(cudaGetLastError() != cudaSuccess) {	\
			printf("cuda error in: %s\n", x);	\
			printf(">>> REACH %s(%s:%d) <<<\n",	\
			       __func__,__FILE__, __LINE__);	\
			exit(0);				\
		}						\
	} while (0)
#endif//_DEBUG

#endif /*GETCUDAERROR_H_*/
