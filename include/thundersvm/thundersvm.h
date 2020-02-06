//
// Created by jiashuai on 17-9-14.
//

#ifndef THUNDERSVM_THUNDERSVM_H
#define THUNDERSVM_THUNDERSVM_H
#include <cstdlib>
#include "util/log.h"
#include <string>
#include <vector>
#include <thundersvm/config.h>
#include "math.h"
#include "util/common.h"
using std::string;
using std::vector;
typedef double float_type;

#ifdef USE_DOUBLE
typedef double kernel_type;
#else
typedef float kernel_type;
#endif
#endif //THUNDERSVM_THUNDERSVM_H
