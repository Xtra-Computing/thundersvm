//
// Created by jiashuai on 17-10-30.
//
#include <thundersvm/model/nusvr.h>
#include <thundersvm/solver/nusmosolver.h>

void NuSVR::train(const DataSet &dataset, SvmParam param) {
    model_setup(dataset, param);
    int n_instances = dataset.n_instances();

    //duplicate instances
    DataSet::node2d instances_2(dataset.instances());
    instances_2.insert(instances_2.end(), dataset.instances().begin(), dataset.instances().end());

    KernelMatrix kernelMatrix(instances_2, param);

    SyncData<real> f_val(n_instances * 2);
    SyncData<int> y(n_instances * 2);

    SyncData<real> alpha_2(n_instances * 2);
    real sum = param.C * param.nu * n_instances / 2;
    for (int i = 0; i < n_instances; ++i) {
        alpha_2[i] = alpha_2[i + n_instances] = min(sum, param.C);
        sum -= alpha_2[i];
        f_val[i] = f_val[i + n_instances] = -dataset.y()[i];
        y[i] = +1;
        y[i + n_instances] = -1;
    }

    int ws_size = min(max2power(n_instances) * 2, 1024);
    NuSMOSolver solver(true);
    solver.solve(kernelMatrix, y, alpha_2, rho[0], f_val, param.epsilon, param.C, param.C, ws_size);
    save_svr_coef(alpha_2, dataset.instances());
}
