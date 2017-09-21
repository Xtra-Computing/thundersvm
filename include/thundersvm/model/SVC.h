//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVC_H
#define THUNDERSVM_SVC_H

#include <thundersvm/kernelmatrix.h>
#include "SvmModel.h"

class SVC: public SvmModel{
public:
    SVC(DataSet &dataSet, const SvmParam &svmParam);

    void train() override;

    void predict(DataSet &dataSet) override;

    void save_to_file(string path) override;

    void load_from_file(string path) override;

private:
    void smo_solver(const KernelMatrix &k_mat, SyncData<int> &y, SyncData<real> &alpha, real &rho, real eps, real C);
};
#endif //THUNDERSVM_SVC_H
