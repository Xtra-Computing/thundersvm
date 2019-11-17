//
// Created by jiashuai on 17-10-5.
//

#ifndef THUNDERSVM_SVR_H
#define THUNDERSVM_SVR_H

#include "thundersvm/thundersvm.h"
#include "svmmodel.h"
#include <map>

using std::map;

/**
 * @brief Support Vector Machine for regression
 */
class SVR : public SvmModel {
public:
    void train(const DataSet &dataset, SvmParam param) override;

    ~SVR() override = default;

protected:
    void model_setup(const DataSet &dataset, SvmParam &param) override;

    /**
     * save \f$\boldsymbel{\alpha}\f$ into coef.
     * @param alpha_2
     * @param instances
     */
    void save_svr_coef(const SyncArray<float_type> &alpha_2, const DataSet::node2d &instances);
};

#endif //THUNDERSVM_SVR_H
