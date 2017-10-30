//
// Created by jiashuai on 17-10-30.
//

#ifndef THUNDERSVM_NUSVR_H
#define THUNDERSVM_NUSVR_H

#include "svr.h"

class NuSVR : public SVR {
public:
    void train(DataSet dataset, SvmParam param) override;

};

#endif //THUNDERSVM_NUSVR_H
