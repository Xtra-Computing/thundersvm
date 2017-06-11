/**
 * trainingDataIO.h
 * Created on: May 21, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef TRAININGDATAIO_H_
#define TRAININGDATAIO_H_

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>

using std::string;
using std::vector;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;

typedef float real;

class CDataIOOps
{
public:
	CDataIOOps(){}
	virtual ~CDataIOOps(){}

	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vData, vector<real> &v_fValue, int, int) = 0;
protected:
	void ReadLibSVMDataFormat(vector<vector<real> > &v_vSample, vector<real> &v_fValue,
										  string strFileName, int nNumofFeatures, int nNumofSamples);
};

/*
 * @brief: a class for reading data of web-a.dst from Microsoft Research
 */
class CReadE2006: public CDataIOOps
{
public:
	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<real> &v_fValue, int, int);
};

class CReadCT: public CDataIOOps
{
public:
	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<real> &v_fValue, int, int);
};

class CReadAMZ: public CDataIOOps
{
public:
	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<real> &v_fValue, int, int);
};

class CReadAbalone: public CDataIOOps
{
public:
	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<real> &v_fValue, int, int);
};

class CReadSlice: public CDataIOOps
{
public:
	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<real> &v_fValue, int, int);
};

class CReadKDD: public CDataIOOps
{
public:
	virtual bool ReadFromFile(string strFileName, vector<vector<real> > &v_vSampleData, vector<real> &v_fValue, int, int);
};


#endif /* TRAININGDATAIO_H_ */
