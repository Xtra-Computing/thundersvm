/*
 * MyAssert.h
 *
 *  Created on: 26 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef MYASSERT_H_
#define MYASSERT_H_

#include <string.h>
#include <assert.h>

//#define _COMPARE_HOST Be careful! Ids of new nodes change now.

#define _DEBUG

#ifndef _DEBUG
	#define PROCESS_ERROR(x)	((void)0)
#else
	#define PROCESS_ERROR(x)	\
		if (!(x))				\
		{						\
			assert(false);		\
		}						\
		NULL;
#endif //_DEBUG



#endif /* MYASSERT_H_ */
