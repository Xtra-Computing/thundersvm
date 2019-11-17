//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_NUSVC_H
#define THUNDERSVM_NUSVC_H

#include "svc.h"

/**
 * @brief Nu-Support Vector Machine for classification
 */
class NuSVC : public SVC {
public:
    ~NuSVC() override = default;

protected:
    void train_binary(const DataSet &dataset, int i, int j, SyncArray<float_type> &alpha, float_type &rho) override;

    void model_setup(const DataSet &dataset, SvmParam &param) override;
};

#endif //THUNDERSVM_NUSVC_H
