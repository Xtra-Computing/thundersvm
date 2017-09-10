
#ifndef BASELIBSVMREADER_H_
#define BASELIBSVMREADER_H_

#include <fstream>
#include "../../SharedUtility/DataType.h"

using std::string;
using std::ifstream;

class BaseLibSVMReader
{
public:
	static void GetDataInfo(string strFileName, int &nNumofFeatures, int &nNumofInstance, uint &nNumofValue);
};



#endif /* BASELIBSVMREADER_H_ */
