//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVC_H
#define THUNDERSVM_SVC_H

#include <map>
#include <thundersvm/kernelmatrix.h>
#include "svmmodel.h"

using std::map;

class SVC : public SvmModel {
public:

    void train(const DataSet &dataset, SvmParam param) override;

    vector<real> predict(const DataSet::node2d &instances, int batch_size) override;

protected:
    virtual void train_binary(const DataSet &dataset, int i, int j, SyncData<real> &alpha, real &rho);

    void model_setup(const DataSet &dataset, SvmParam &param) override;

private:

    vector<real> predict_label(const SyncData<real> &dec_values, int n_instances) const;

    void probability_train(const DataSet &dataset);

    void multiclass_probability(const vector<vector<real>> &r, vector<real> &p) const;

    vector<real> c_weight;


};

#endif //THUNDERSVM_SVC_H
