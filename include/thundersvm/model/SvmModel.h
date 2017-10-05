//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMMODEL_H
#define THUNDERSVM_SVMMODEL_H

#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <thundersvm/kernelmatrix.h>

class SvmModel{
public:
    SvmModel(DataSet &dataSet, const SvmParam &svmParam);
    virtual void train() = 0;

    virtual vector<real> predict(const DataSet::node2d &instances, int batch_size) = 0;
    virtual void save_to_file(string path) = 0;
    virtual void load_from_file(string path) = 0;

protected:
    int max2power(int n) const;
    void smo_solver(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
                    SyncData<real> &init_f, real eps, real C, int ws_size);
    DataSet &dataSet;
    SvmParam svmParam;
};
#endif //THUNDERSVM_SVMMODEL_H
