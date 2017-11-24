/*
 * cmdparser.h
 *
 *  Created on: Oct 14, 2017
 *      Author: Zeyi Wen
 */

#ifndef CMDPARSER_H_
#define CMDPARSER_H_

#include "svmparam.h"

/**
 * @brief Command-line parser
 */
class CMDParser{
public:
	CMDParser() : do_cross_validation(false), nr_fold(0), gpu_id(0) {};

	void parse_command_line(int argc, char **argv);

	void parse_python(int argc, char **argv);

	SvmParam param_cmd;
	bool do_cross_validation;
	int nr_fold;
	int gpu_id;
	char svmtrain_input_file_name[1024];
	char svmpredict_input_file[1024];
	char svmpredict_output_file[1024];
	char svmpredict_model_file_name[1024];
	char model_file_name[1024];
};



#endif /* CMDPARSER_H_ */
