/*
 * cmdparser.h
 *
 *  Created on: Oct 14, 2017
 *      Author: Zeyi Wen
 */

#ifndef CMDPARSER_H_
#define CMDPARSER_H_

#include "thundersvm.h"
#include "svmparam.h"

/**
 * @brief Command-line parser
 */
class CMDParser{
public:
    CMDParser() : do_cross_validation(false), gamma_set(false), nr_fold(0), gpu_id(0), n_cores(-1) {};

	void parse_command_line(int argc, char **argv);

	void parse_python(int argc, char **argv);

    bool check_parameter();

	SvmParam param_cmd;
	bool do_cross_validation;
    bool gamma_set;
	int nr_fold;
	int gpu_id;
	int n_cores;
    string svmtrain_input_file_name;
    string svmpredict_input_file;
    string svmpredict_output_file;
    string svmpredict_model_file_name;
    string model_file_name;
};



#endif /* CMDPARSER_H_ */
