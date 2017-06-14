/*
 * binarySearch.cu
 *
 *  Created on: Jun 12, 2017
 *      Author: zeyi
 */

#include "binarySearch.h"

__device__ void RangeBinarySearch(uint pos, const uint* pSegStartPos, uint numSeg, uint &segId)
{
	uint midSegId;
	uint startSegId = 0, endSegId = numSeg - 1;
	segId = -1;
	while(startSegId <= endSegId){
		midSegId = startSegId + ((endSegId - startSegId) >> 1);//get the middle index
		if(pos >= pSegStartPos[midSegId] && (midSegId == endSegId || pos < pSegStartPos[midSegId + 1]))
		{
			segId = midSegId;
			return;
		}
		else if(pos >= pSegStartPos[midSegId + 1])
			startSegId = midSegId + 1;//find left part
		else
			endSegId = midSegId - 1;//find right part
	}
}
