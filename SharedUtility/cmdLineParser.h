/*
 * commandLineParser.h
 *
 *  Created on: Jun 14, 2017
 *      Author: zeyi
 */

#ifndef COMMANDLINEPARSER_H_
#define COMMANDLINEPARSER_H_


class CmdLineParser
{
public:
	static int numFeature;

public:
	CmdLineParser(){}
	virtual~CmdLineParser(){}

	void ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName);
	virtual bool HandleOption(char c, char *pcOptionValue) = 0;
	virtual void HelpInfo() = 0;
	virtual void InitParam(){}
};



#endif /* COMMANDLINEPARSER_H_ */
