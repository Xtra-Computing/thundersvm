/*
 * commandLineParser.h
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#ifndef COMMANDLINEPARSER_H_
#define COMMANDLINEPARSER_H_

#include "../svm-shared/SVMParam.h"

class Parser
{
public:
	static int task_type;
	static bool compute_training_error;
	static int nr_fold;
	static int nNumofFeature;
	static SVMParam param;
public:
	static void ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName);
	static void HelpInfo();
};


#endif /* COMMANDLINEPARSER_H_ */
