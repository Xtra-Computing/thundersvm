//
// Created by jiashuai on 17-10-6.
//

#ifndef THUNDERSVM_ONECLASS_SVC_H
#define THUNDERSVM_ONECLASS_SVC_H

#include "svmmodel.h"

class OneClassSVC : public SvmModel {
public:
    OneClassSVC(DataSet &dataSet, const SvmParam &svmParam);

    void train() override;

    void save_to_file(string path) override;

    void load_from_file(string path) override;

};

#endif //THUNDERSVM_ONECLASS_SVC_H
