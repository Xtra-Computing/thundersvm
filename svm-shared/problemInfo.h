
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
