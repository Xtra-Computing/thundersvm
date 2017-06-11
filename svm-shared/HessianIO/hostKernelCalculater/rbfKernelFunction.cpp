/*
 * rbfKernelFunction.cpp
 *
 *  Created on: 29/10/2015
 *      Author: Zeyi Wen
 */

#include <assert.h>
#include <math.h>
#include <iostream>
#include "rbfKernelFunction.h"

using std::cout;
using std::endl;

void RBFKernelFunction::ComputeRow(vector<vector<float> > &vvDocVector, int rowId, int nNumofRow, float *pRow) {
    assert(rowId >= 0 && nNumofRow > 0 && pRow != NULL);
    int numofInstance = vvDocVector.size();
    assert(numofInstance > 0);
    for (int j = 0; j < nNumofRow; j++) {//compute row of index at rowId
//		cout << "computing " << j << " row" << endl;
        for (int i = 0; i < numofInstance; i++) {
            float value_ij = RBF(vvDocVector[j + rowId], vvDocVector[i]);

            pRow[i + j * numofInstance] = value_ij;
        }
    }
}

/*
 * @brief: compute the kernel value of two instances
 */
float RBFKernelFunction::RBF(vector<float> &ins1, vector<float> &ins2) {
    float value = 0;

    value = exp(-m_gamma * (square(ins1) + square(ins2) - 2 * dotProduct(ins1, ins2)));

    return value;
}

void RBFKernelFunction::ComputeSparseRow(vector<vector<svm_node> > &v_v_DocVector, int rowId, int nNumofRow,
                                         float *pRow) {
    for (int i = 0; i < nNumofRow; ++i) {
        for (int j = 0; j < v_v_DocVector.size(); ++j) {
            vector<svm_node> x = v_v_DocVector[rowId + i];
            vector<svm_node> y = v_v_DocVector[j];
            real xDot = 0;
            real yDot = 0;
            for (int k = 0; k < x.size() - 1; ++k) {
                xDot += x[k].value * x[k].value;
            }
            for (int k = 0; k < y.size() - 1; ++k) {
                yDot += y[k].value * y[k].value;
            }
            int ix = 0;
            int iy = 0;
            float dot = 0;
            while (x[ix].index != -1 && y[iy].index != -1) {
                if (x[ix].index == y[iy].index)
                    dot += x[ix++].value * y[iy++].value;
                else if (x[ix].index > y[iy].index)
                    iy++;
                else
                    ix++;
            }
            pRow[i * v_v_DocVector.size() + j] = expf(-m_gamma * (xDot + yDot - 2 * dot));
        }
    }
}
