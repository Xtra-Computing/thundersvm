
#include "fileOps.h"
#include<cstdlib>
//#include <boost/interprocess/shared_memory_object.hpp>
//#include <boost/interprocess/file_mapping.hpp>

/*
 * @brief: write a few Hessian rows to file at one time
 * @return: a set of starting positions for Hessian rows in file
 */
bool CFileOps::WriteToFile(ofstream &writeOut, real *pContent, int nNumofRows, int nNumofColumns)
{
	bool bReturn = false;
	if(!writeOut.is_open() || pContent == NULL || nNumofRows <= 0 || nNumofColumns <= 0)
	{
		cerr << "write rows to file failed: invalid input param" << endl;
		return bReturn;
	}

	//allocate memory for remembering locations of these rows
	//bReturn = new int[nNumofRows];
//	for(int i = 0; i < nNumofRows; i++)
	{
		//write one Hessian row to file
		//pnReturn[i] =
		int nWriteResult = WriteToFile(writeOut, pContent, nNumofColumns * nNumofRows);
		if(nWriteResult == -1)
		{
			cerr << "write to file error" << endl;
			exit(1);
		}
		//if(pnReturn[i] == -1)
		//{
		//	cerr << "error in WriteToFile: position is -1" << endl;
		//	delete[] bReturn;
		//	pnReturn = NULL;
		//	break;
		//}
//		pContent += nNumofColumns;
	}
	bReturn = true;

	return bReturn;
}

/*
 * @brief: write one Hessian row to file
 * @return: the starting position of this Hessian row
 */
int CFileOps::WriteToFile(ofstream &writeOut, real *pContent, int nSizeofContent)
{
	int nReturn = -1;
	if(!writeOut.is_open() || pContent == NULL || nSizeofContent <= 0)
	{
		cerr << "write content to file failed: input param invalid" << endl;
		return nReturn;
	}

	//return the starting postion of this Hessian row
	nReturn = writeOut.tellp();
	//for(int i = 0; i < nSizeofContent; i++)
	//{
		writeOut.write((char*)pContent, sizeof(real) * nSizeofContent);
		//writeOut /*<< setprecision(16)*/ << (float_point)*pContent;
		//if(i != nSizeofContent - 1)
		//	writeOut << " : ";
	//	pContent++;
	//}
	//writeOut << endl;

	return nReturn;
}

/*
 * @brief: read a Hessian row from file
 * @param: nIndexofRow : the row to be read
 * @param: nNumofElementsPerRow: the # of elements of a Hessian row
 * @return: true if success
 */
bool CFileOps::ReadRowsFromFile(FILE *&readIn, real *&pContent, const int &nNumofElementsPerRow, int nNumofRowsToRead,
							const int &nIndexofRow)
{
	bool bReturn = false;
	assert(readIn != NULL && pContent != NULL && nNumofElementsPerRow > 0 && nIndexofRow >= 0);
	//find the position of this Hessian row
	long long nSeekPos = sizeof(real) * nIndexofRow * (long long)nNumofElementsPerRow;

	//cout << nIndexofRow << " v.s. " << nSeekPos << " v.s " << sizeof(pContent)<< endl;
	fseek(readIn, nSeekPos, SEEK_SET);
	assert(ftell(readIn) != -1);

	//cout << nNumofElementsPerRow << " v.s " << nNumofRowsToRead << endl;
	long nNumofRead = fread(pContent, sizeof(real), nNumofElementsPerRow * nNumofRowsToRead, readIn);
	assert(nNumofRead > 0);

	//clean eof bit, when pointer reaches end of file
	if(feof(readIn))
	{
		rewind(readIn);
	}
	bReturn = true;

	return bReturn;
}

//bool CFileOps::ReadPartOfRowFromFile(boost::interprocess::mapped_region *pRegion, float_point *pContent, int nFullRowSize, int nNumofElementsToRead, long long nIndexof1stElement)
//{
	//bool bReturn = false;

	//if(pContent == NULL || nFullRowSize <= 0 || nNumofElementsToRead <= 0 || nIndexof1stElement < 0)
	//{
		//cerr << "error in ReadPartOfRowFromFile: invalid param" << endl;
		//return bReturn;
	//}

	//float_point *pStartPos = static_cast<float_point*>(pRegion->get_address());
	////find the position of this Hessian row
	//pStartPos = pStartPos + nIndexof1stElement;
	//memcpy(pContent, pStartPos, sizeof(float_point) * nNumofElementsToRead);

	//bReturn = true;
	//return bReturn;
//}
