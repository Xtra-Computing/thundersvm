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

    virtual vector<float_type> predict(const DataSet::node2d &instances, int batch_size);

    void predict_dec_values(const DataSet::node2d &instances, SyncData<float_type> &dec_values, int batch_size) const;

    virtual vector<float_type> cross_validation(DataSet dataset, SvmParam param, int n_fold);

    virtual void save_to_file(string path);

    virtual void load_from_file(string path);

protected:

    virtual void model_setup(const DataSet &dataset, SvmParam &param);


    SvmParam param;
    SyncData<float_type> coef;
    DataSet::node2d sv;
    SyncData<int> n_sv;//the number of sv in each class
    SyncData<float_type> rho;
    int n_classes = 2;
    size_t n_binary_models;
    int n_total_sv;
    vector<float_type> probA;
    vector<float_type> probB;

    //for classification
    vector<int> label;
};

#endif //THUNDERSVM_SVMMODEL_H
