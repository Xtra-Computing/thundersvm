/*
 * cmdLineParser.cpp
 *
 *  Created on: Jun 14, 2017
 *      Author: zeyi
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "cmdLineParser.h"

int CmdLineParser::numFeature = 0;

void CmdLineParser::ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName){
		int i;
		void (*print_func)(const char*) = NULL;	// default printing to stdout

		InitParam();

		// parse options
		for(i=1;i<argc;i++)
		{
			if(argv[i][0] != '-') break;
			if(++i>=argc)
				HelpInfo();
			switch(argv[i-1][1])
			{
				case 'f':
					numFeature = atoi(argv[i]);
					if(numFeature < 1)
						HelpInfo();
					break;

				default:
					if(HandleOption(argv[i-1][1], argv[i]) == true)
						break;
					fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
					HelpInfo();
			}
		}

		if(i>=argc)
			HelpInfo();

		strcpy(pcFileName, argv[i]);

		if(i<argc-1)
			strcpy(pcSavedFileName,argv[i+1]);
		else
		{
			char *p = strrchr(argv[i],'/');
			if(p==NULL)
				p = argv[i];
			else
				++p;
			sprintf(pcSavedFileName,"%s.model",p);
		}
}
