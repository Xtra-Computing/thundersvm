//
// Created by jiashuai on 17-10-6.
//

#include <thundersvm/model/oneclass_svc.h>

OneClassSVC::OneClassSVC(DataSet &dataSet, const SvmParam &svmParam) : SvmModel(dataSet, svmParam) {
    n_instances = dataSet.instances().size();
}

void OneClassSVC::train() {
    SyncData<real> alpha(n_instances);
    SyncData<real> init_f(n_instances);

    KernelMatrix kernelMatrix(dataSet.instances(), dataSet.n_features(), svmParam.gamma);
}

vector<real> OneClassSVC::predict(const DataSet::node2d &instances, int batch_size) {
}

void OneClassSVC::save_to_file(string path) {

}

void OneClassSVC::load_from_file(string path) {

}
