//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMMODEL_H
#define THUNDERSVM_SVMMODEL_H

#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <thundersvm/kernelmatrix.h>
#include <map>

using std::map;

class SvmModel {
public:
    virtual void train(const DataSet &dataset, SvmParam param) = 0;

    virtual vector<real> predict(const DataSet::node2d &instances, int batch_size);

    virtual vector<real> cross_validation(DataSet dataset, SvmParam param, int n_fold);

    virtual void save_to_file(string path);

    virtual void load_from_file(string path);

protected:

    virtual void model_setup(const DataSet &dataset, SvmParam &param);

    void predict_dec_values(const DataSet::node2d &instances, SyncData<real> &dec_values, int batch_size) const;

    SvmParam param;
    SyncData<real> coef;
    DataSet::node2d sv;
    SyncData<int> n_sv;//the number of sv in each class
    SyncData<real> rho;
    int n_classes = 2;
    size_t n_binary_models;
    int n_total_sv;
    vector<real> probA;
    vector<real> probB;

    //for classification
    vector<int> label;
};

#endif //THUNDERSVM_SVMMODEL_H
