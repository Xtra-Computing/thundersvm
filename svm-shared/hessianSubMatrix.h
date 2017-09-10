
#ifndef HESSIAN_SUB_MATRIX_H
#define HESSIAN_SUB_MATRIX_H

class SubMatrix
{
public:
	int nRowIndex;	//the row index of this sub matrix in global hessian matrix
	int nColIndex;	//the column index in global hessian matrix
	int nRowSize;	//the number of rows of this sub matrix
	int nColSize;	//the number of column of this sub matrix
public:
	bool isValid()
	{
		if(nRowIndex >= 0 && nColIndex >= 0 && nRowSize > 0 && nColSize > 0)
			return true;
		return false;
	}
};

#endif //HESSIAN_SUB_MATRIX_H
