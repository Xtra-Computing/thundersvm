/*
 * problemInfo.h
 *
 *  Created on: 28/10/2015
 *      Author: Zeyi Wen
 */

#ifndef PROBLEMINFO_H_
#define PROBLEMINFO_H_

#include <iostream>

using std::string;

struct problemInfo{
	int dim;
	int numofIns;
	float c;
	float gamma;
	string fileName;

	void InitDate(problemInfo *dataset, int numofDataset);
};


#endif /* PROBLEMINFO_H_ */
