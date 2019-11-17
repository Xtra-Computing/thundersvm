//
// Created by jiashuai on 17-10-30.
//

#ifndef THUNDERSVM_NUSVR_H
#define THUNDERSVM_NUSVR_H

#include "svr.h"

/**
 * @brief Support Vector Machine for regression
 */
class NuSVR : public SVR {
public:
    void train(const DataSet &dataset, SvmParam param) override;

    ~NuSVR() override = default;

protected:
    void model_setup(const DataSet &dataset, SvmParam &param) override;

};

#endif //THUNDERSVM_NUSVR_H
