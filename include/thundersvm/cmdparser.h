/*
 * cmdparser.h
 *
 *  Created on: Oct 14, 2017
 *      Author: Zeyi Wen
 */

#ifndef CMDPARSER_H_
#define CMDPARSER_H_

#include "svmparam.h"

class CMDParser{
public:
	void parse_command_line(int argc, char **argv);
	void init_param();

	struct SvmParam param_cmd;
	char svmtrain_input_file_name[1024];
	char model_file_name[1024];
};



#endif /* CMDPARSER_H_ */
