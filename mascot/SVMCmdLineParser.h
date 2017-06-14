/*
 * commandLineParser.h
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#ifndef SVMCOMMANDLINEPARSER_H_
#define SVMCOMMANDLINEPARSER_H_

#include <iostream>
#include "../svm-shared/svmParam.h"
#include "../SharedUtility/cmdLineParser.h"

using std::string;

class SVMCmdLineParser:public CmdLineParser
{
public:
	static int task_type;
	static bool compute_training_error;
	static int nr_fold;
	static SVMParam param;
	static string testSetName;
public:
	SVMCmdLineParser(){}
	virtual ~SVMCmdLineParser(){}
	virtual bool HandleOption(char c, char *pcOptionValue);
	virtual void HelpInfo();
	virtual void InitParam();
};


#endif /* SVMCOMMANDLINEPARSER_H_ */
