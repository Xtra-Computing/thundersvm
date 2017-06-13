/*
 * commandLineParser.h
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#ifndef COMMANDLINEPARSER_H_
#define COMMANDLINEPARSER_H_

#include <iostream>
#include "../svm-shared/svmParam.h"

using std::string;

class Parser
{
public:
	static int task_type;
	static bool compute_training_error;
	static int nr_fold;
	static int numFeature;
	static SVMParam param;
	static string testSetName;
public:
	static void ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName);
	static void HelpInfo();
};


#endif /* COMMANDLINEPARSER_H_ */
