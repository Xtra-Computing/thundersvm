/**
 * fileOps.h
 * Created on: May 22, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef FILEOPS_H_
#define FILEOPS_H_

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <boost/interprocess/mapped_region.hpp>

#include "gpu_global_utility.h"

using std::string;
using std::fstream;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;
using std::endl;
using std::cout;
using std::setprecision;

class CFileOps
{
public:
	static bool WriteToFile(ofstream &writeOut, float_point *pContent, int nNumofRows, int nNumofColumns);
	static int WriteToFile(ofstream &writeOut, float_point *pContent, int nSizeofContent);

	static bool ReadRowsFromFile(FILE *&readIn, float_point *&pContent, const int &nNumofElementsPerRow,
						  int nNumofRowsToRead, const int &nIndexofRow);
	static bool ReadPartOfRowFromFile(boost::interprocess::mapped_region*, float_point *pContent, int nFullRowSize, int nNumofElementsToRead, long long nIndexof1stElement);

	/*
	 * @brief: read a continuous part from a file
	 * @param: pContent: storing the read content
	 * @param: nFullRowSize: the size of a full row in hessian matrix because we store the whole hessian matrix
	 * @param: nNumofElementsToRead: the number of elements read by this function
	 * @param: nIndexof1stElement: the start point of this reading procedure.
	 */
	void ReadPartOfRowFromFile(FILE *&readIn, float_point *pContent, int nNumofElementsToRead, long long nIndexof1stElement)
	{
		//bool bReturn = false;

		assert(readIn != NULL && pContent != NULL && nNumofElementsToRead > 0 && nIndexof1stElement >= 0);
		//find the position of this Hessian row
		fseek(readIn, 0, SEEK_END);
		assert(ftell(readIn) != 0);

		long long nSeekPos = sizeof(float_point) * nIndexof1stElement;
		fseek(readIn, nSeekPos, SEEK_SET);
//		cout << ftell(readIn) << endl;
		assert(ftell(readIn) != -1);

		int nNumofRead = fread(pContent, sizeof(float_point), nNumofElementsToRead, readIn);

//		cout << ftell(readIn) << endl;
//		assert(nNumofRead > 0);
//		cout << "the number of kernel values read " << nNumofRead << endl;
		if(ferror(readIn) == true)
		{
			cout  << "read kernel values from file error" << endl;
		}

		//clean eof bit, when pointer reaches end of file
		if(feof(readIn))
		{
//			cout << "end of file is reached" << endl;
			rewind(readIn);
		}
	}
};


#endif /* FILEOPS_H_ */
