//
// Created by jiashuai on 17-10-25.
//

#ifndef THUNDERSVM_NUSVC_H
#define THUNDERSVM_NUSVC_H

#include "svc.h"

class NuSVC : public SVC {
protected:
    void train_binary(const DataSet &dataset, int i, int j, int k) override;
};

#endif //THUNDERSVM_NUSVC_H
