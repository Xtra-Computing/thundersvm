//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVC_H
#define THUNDERSVM_SVC_H

#include <map>
#include <thundersvm/kernelmatrix.h>
#include "svmmodel.h"

using std::map;

/**
 * @brief Support Vector Machine for classification
 */
class SVC : public SvmModel {
public:

    void train(const DataSet &dataset, SvmParam param) override;

    vector<float_type> predict(const DataSet::node2d &instances, int batch_size) override;

    ~SVC() override = default;
protected:
    /**
     * train a binary SVC model \f$SVM_{i,j}\f$ for class i and class j.
     * @param [in] dataset original dataset
     * @param [in] i
     * @param [in] j
     * @param [out] alpha optimization variables \f$\boldsymbol{\alpha}\f$ in dual problem, should be initialized with
     * the same size of the number
     * of instances in this binary problem
     * @param [out] rho bias term \f$b\f$ in dual problem
     */
    virtual void train_binary(const DataSet &dataset, int i, int j, SyncArray<float_type> &alpha, float_type &rho);

    void model_setup(const DataSet &dataset, SvmParam &param) override;

private:

    /**
     * predict final labels using voting. \f$SVM_{i,j}\f$ will vote for class i if its decision value greater than 0,
     * otherwise it will vote for class j. The final label will be the one with the most votes.
     * @param dec_values decision values for each instance and each binary model
     * @param n_instances the number of instances to be predicted
     * @return final labels of each instances
     */
    vector<float_type> predict_label(const SyncArray<float_type> &dec_values, int n_instances) ;

    /**
     * perform probability training.
     * If param.probability equals to 1, probability_train() will be called to train probA and probB and the model will
     * be able to produce probability outputs.
     * @param dataset training dataset, should be already grouped using group_classes().
     */
    void probability_train(const DataSet &dataset);

    /**
     * transform n_binary_models binary probabilities in to probabilities of each class
     * @param [in] r n_binary_models binary probabilities
     * @param [out] p probabilities for each class
     */
    void multiclass_probability(const vector<vector<float_type>> &r, vector<float_type> &p) const;

    /**
     * class weight for each class, the final \f$C_{i}\f$ of class i will be \f$C*\text{c_weight}[i]\f$
     */
    vector<float_type> c_weight;


};

#endif //THUNDERSVM_SVC_H
