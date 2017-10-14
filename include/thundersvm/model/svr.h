//
// Created by jiashuai on 17-10-5.
//

#ifndef THUNDERSVM_SVR_H
#define THUNDERSVM_SVR_H

#include "thundersvm/thundersvm.h"
#include "svmmodel.h"
#include <map>

using std::map;

class SVR : public SvmModel {
public:

    SVR() : SvmModel() {
        param.svm_type = SvmParam::EPSILON_SVR;
    }

    void train(DataSet dataset, SvmParam param) override;

    void save_to_file(string path) override;

    void load_from_file(string path) override;
};

#endif //THUNDERSVM_SVR_H
