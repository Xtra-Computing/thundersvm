/*
 * problemInfo.cpp
 *
 *  Created on: 31/10/2015
 *      Author: Zeyi Wen
 */

#include "problemInfo.h"

void problemInfo::InitDate(problemInfo *svrData, int numofDataset)
{
	svrData[0].c = 32;
	svrData[0].gamma = 0.0625;
	svrData[0].fileName = "dataset/normalized_amz.txt";

	svrData[1].c = 64;
	svrData[1].gamma = 0.25;
	svrData[1].fileName = "dataset/slice_loc.txt";

	svrData[2].c = 256;
	svrData[2].gamma = 0.125;
	svrData[2].fileName = "dataset/E2006.train";

	svrData[3].c = 64;
	svrData[3].gamma = 0.25;
	svrData[3].fileName = "dataset/kdd98.txt";


	/*//full datasets
	{
		svrData[0].numofIns = 30000;
		svrData[0].dim = 20000;

		svrData[1].numofIns = 53500;
		svrData[1].dim = 386;

		svrData[2].numofIns = 16087;
		svrData[2].dim = 150360;

		svrData[3].numofIns = 191779;
		svrData[3].dim = 479;
	}*/

	//sub-datasets
	{
		svrData[0].numofIns = 15000;
		svrData[0].dim = 3500;

		svrData[1].numofIns = 53500;
		svrData[1].dim = 386;

		svrData[2].numofIns = 16087;
		svrData[2].dim = 3500;

		svrData[3].numofIns = 50000;
		svrData[3].dim = 479;
	}
}

