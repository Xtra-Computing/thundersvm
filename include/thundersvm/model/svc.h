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

    vector<float_type> predict(const DataSet::node2d &instances, int batch_size) override;

protected:
    virtual void train_binary(const DataSet &dataset, int i, int j, SyncData<float_type> &alpha, float_type &rho);

    void model_setup(const DataSet &dataset, SvmParam &param) override;

private:

    vector<float_type> predict_label(const SyncData<float_type> &dec_values, int n_instances) const;

    void probability_train(const DataSet &dataset);

    void multiclass_probability(const vector<vector<float_type>> &r, vector<float_type> &p) const;

    vector<float_type> c_weight;


};

#endif //THUNDERSVM_SVC_H
