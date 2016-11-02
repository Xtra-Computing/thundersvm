//
// Created by ss on 16-11-2.
//

#ifndef MASCOT_SVM_SVMMODEL_H
#define MASCOT_SVM_SVMMODEL_H

#include<vector>
#include "../svm-shared/gpu_global_utility.h"
#include "svmProblem.h"

using std::vector;

class svmModel {
private:
    unsigned int nrClass;
    unsigned int cnr2;
    vector<vector<vector<float_point> > > supportVectors;
    vector<vector<float_point> > coef;
    vector<float_point> rho;
    vector<float_point> probA;
    vector<float_point> probB;
    vector<float_point> label;

    unsigned int getK(int i, int j)const;

public:
    svmModel(unsigned int nrClass) : nrClass(nrClass) {
        cnr2 = (nrClass) * (nrClass - 1) / 2;
        coef.resize(cnr2);
        rho.resize(cnr2);
        probA.resize(cnr2);
        probB.resize(cnr2);
        supportVectors.resize(nrClass);
    }

    void addBinaryModel(const svmProblem& ,const svm_model&, int i, int j);

    float_point getRho(int i, int j) const;
};


#endif //MASCOT_SVM_SVMMODEL_H
