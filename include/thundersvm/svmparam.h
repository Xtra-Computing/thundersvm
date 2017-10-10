//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_SVMPARAM_H
#define THUNDERSVM_SVMPARAM_H

#include "thundersvm.h"

struct SvmParam{
    real C;
    real gamma;
    real p;
    real nu;
    real epsilon;
};
#endif //THUNDERSVM_SVMPARAM_H
