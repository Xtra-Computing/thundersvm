//
// Created by ss on 16-11-15.
//

#ifndef MASCOT_SVM_HOSTHESSIANONFLY_H
#define MASCOT_SVM_HOSTHESSIANONFLY_H


#include "baseHessian.h"
#include "hostKernelCalculater/kernelFunction.h"
#include "hostKernelCalculater/rbfKernelFunction.h"
#include"cuda_runtime.h"
#include"helper_cuda.h"
#include "../gpu_global_utility.h"

class HostHessianOnFly : public BaseHessian {
public:
    HostHessianOnFly(KernelFunction &function, vector<vector<svm_node> > &samples) :
            kernelCalculator(function), samples(samples) {};

    virtual void ReadRow(int nPosofRowAtHessian, float_point *pfHessianRow) override;

    virtual bool PrecomputeHessian(const string &strHessianMatrixFileName, const string &strDiagHessianFileName,
                                   vector<vector<float_point> > &v_v_DocVector) override;

    virtual bool
    GetHessianDiag(const string &strFileName, const int &nNumofTraingSamples, float_point *pfHessianDiag) override;

//    virtual bool AllocateBuffer(int nNumofRows) override;

//    virtual bool ReleaseBuffer() override;

private:
    vector<vector<svm_node> > &samples;
    KernelFunction &kernelCalculator;
};


#endif //MASCOT_SVM_HOSTHESSIANONFLY_H
