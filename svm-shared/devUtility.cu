#include "devUtility.h"

__device__ void GetMinValueOriginal(float_point *pfValues, int *pnKey, int nNumofBlock)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64)
		{
			if(compOffset < nNumofBlock)
			{
				if(pfValues[compOffset] < pfValues[nTid])
				{
					pnKey[nTid] = pnKey[compOffset];
					pfValues[nTid] = pfValues[compOffset];
				}
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = pfValues[compOffset];
			}
		}

}

__device__ void GetMinValueOriginal(float_point *pfValues, int nNumofBlock)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}


		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			if(pfValues[compOffset] < pfValues[nTid])
			{
				pfValues[nTid] = pfValues[compOffset];
			}
		}

}


/* *
 /*
 * @brief: use reducer to get the minimun value in parallel
 * @param: pfValues: a pointer to a set of data
 * @param: pnKey:	 a pointer to the index of the set of data. It's for getting the location of min.
 */
__device__ void GetMinValue(float_point *pfValues, int *pnKey, int nNumofBlock)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;
	float_point fValue1, fValue2;
	fValue1 = pfValues[nTid];

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64)
		{
			if(compOffset < nNumofBlock)
			{
				fValue2 = pfValues[compOffset];
				if(fValue2 < fValue1)
				{
					pnKey[nTid] = pnKey[compOffset];
					pfValues[nTid] = fValue2;
					fValue1 = fValue2;
				}
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				fValue1 = fValue2;
			}
		}

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			if(fValue2 < fValue1)
			{
				pnKey[nTid] = pnKey[compOffset];
				pfValues[nTid] = fValue2;
				//fValue1 = fValue2;
			}
		}
}

__device__ void GetMinValue(float_point *pfValues, int nNumofBlock)
{
	/*if(1024 < BLOCK_SIZE)
	{
		printf("block size is two large!\n");
		return;
	}*/
	//Reduce by a factor of 2, and minimize step size
	int nTid = threadIdx.x;
	int compOffset;
	float_point fValue1, fValue2;
	fValue1 = pfValues[nTid];

	if(BLOCK_SIZE == 128)
	{
		compOffset = nTid + 64;
		if(nTid < 64 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();
	}
		compOffset = nTid + 32;
		if(nTid < 32 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}
		//synchronise threads to avoid read dirty value (dirty read may happen if two steps reduction, say 32 and 16, run simultaneously)
		__syncthreads();

		compOffset = nTid + 16;
		if(nTid < 16 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}

		compOffset = nTid + 8;
		if(nTid < 8 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}

		compOffset = nTid + 4;
		if(nTid < 4 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}


		compOffset = nTid + 2;
		if(nTid < 2 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			fValue1 = (fValue2 < fValue1) ? fValue2 : fValue1;
			pfValues[nTid] = fValue1;
		}

		compOffset = nTid + 1;
		if(nTid < 1 && (compOffset < nNumofBlock))
		{
			fValue2 = pfValues[compOffset];
			pfValues[nTid] = (fValue2 < fValue1) ? fValue2 : fValue1;
		}

}
