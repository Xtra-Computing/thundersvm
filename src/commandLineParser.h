/*
 * commandLineParser.h
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#ifndef COMMANDLINEPARSER_H_
#define COMMANDLINEPARSER_H_

class Parser
{
public:
	static int cross_validation;
	static int nr_fold;
	static int nNumofFeature;
	static SVMParam param;
public:
	static void ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName);
	static void HelpInfo();
};


#endif /* COMMANDLINEPARSER_H_ */
