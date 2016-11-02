//
// Created by ss on 16-11-2.
//

#include <cstdio>
#include "svmModel.h"

unsigned int svmModel::getK(int i, int j) const{
    return ((nrClass - 1) + (nrClass - i)) * i / 2 + j - i - 1;
}

void svmModel::addBinaryModel(const svmProblem &problem, const svm_model &bModel, int i, int j) {
    unsigned int k = getK(i, j);
    for (int l = 0; l < bModel.nSV[0] + bModel.nSV[1]; ++l) {
//        printf("%d:%f.2|",bModel.pnIndexofSV[l],bModel.sv_coef[0][l]);
        coef[k].push_back(bModel.sv_coef[0][l]);
        for (int m = 0; m < problem.getNumOfFeatures(); ++m) {
            supportVectors[k].push_back(problem.v_vSamples[bModel.pnIndexofSV[l]]);
        }
//        printf("%d|",problem.v_nLabels[bModel.pnIndexofSV[l]]);
    }
//    printf("\n");
    rho[k] = bModel.rho[0];
}

float_point svmModel::getRho(int i, int j) const {
    return rho[getK(i,j)];
}
