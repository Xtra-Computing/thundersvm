//
// Created by ss on 16-11-15.
//

#ifndef MASCOT_SVM_HOSTHESSIANONFLY_H
#define MASCOT_SVM_HOSTHESSIANONFLY_H


#include "baseHessian.h"
#include "hostKernelCalculater/kernelFunction.h"
#include "hostKernelCalculater/rbfKernelFunction.h"

class HostHessianOnFly : public BaseHessian {
public:
    HostHessianOnFly(KernelFunction &function, vector<vector<float_point> > &samples) :
            kernelCalculator(function), samples(samples) {};

    virtual void ReadRow(int nPosofRowAtHessian, float_point *pfHessianRow) override;

    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<float_point> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag) override;

private:
    vector<vector<float_point> > &samples;
    KernelFunction &kernelCalculator;
};


#endif //MASCOT_SVM_HOSTHESSIANONFLY_H
