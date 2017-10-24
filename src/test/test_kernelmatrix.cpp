//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"
#include <dataset.h>
real rbf_kernel(const DataSet::node2d &instances, int x, int y, real gamma) {
    real sum = 0;
    auto i = instances[x].begin();
    auto j = instances[y].begin();
    while (i != instances[x].end() && j != instances[y].end()) {
        if (i->index < j->index) {
            sum += i->value * i->value;
            i++;
        } else if (i->index > j->index) {
            sum += j->value * j->value;
            j++;
        } else {
            sum += (i->value - j->value) * (i->value - j->value);
            i++;
            j++;
        }
    }
    while (i != instances[x].end()) {
        sum += i->value * i->value;
        i++;
    }
    while (j != instances[y].end()) {
        sum += j->value * j->value;
        j++;
    }
    return expf(-gamma * sum);
}

real dot_product(const DataSet::node2d &instances, int x, int y) {
    real sum = 0;
    auto i = instances[x].begin();
    auto j = instances[y].begin();
    while (i != instances[x].end() && j != instances[y].end()) {
        if (i->index < j->index) {
            i++;
        } else if (i->index > j->index) {
            j++;
        } else {
            sum += i->value * j->value;
            i++;
            j++;
        }
    }
    return sum;
}

TEST(KernelMatrixTest, get_rows) {
    DataSet dataSet;
    dataSet.load_from_file(DATASET_DIR "test_dataset.txt");
    unsigned n_rows = 40;
    SvmParam param;
    SyncData<int> rows(n_rows);
    SyncData<real> kernel_rows(n_rows * dataSet.total_count());
    for (unsigned i = 0; i < n_rows; ++i) {
        rows.host_data()[i] = i*3 + 4;
    }

    //test rbf kernel
    param.gamma = 0.5;
    param.kernel_type = SvmParam::RBF;
    KernelMatrix *kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    DataSet::node2d instances(dataSet.instances().begin(), dataSet.instances().begin() + n_rows);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        real cpu_kernel = rbf_kernel(dataSet.instances(), i, i, param.gamma);
        real gpu_kernel = kernelMatrix->diag()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = rbf_kernel(dataSet.instances(), rows[i], j, param.gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << rows[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = rbf_kernel(dataSet.instances(), i, j, param.gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i << "," << j;
        }
    }
    delete kernelMatrix;

//    //test poly kernel
    param.gamma = 0.5;
    param.degree = 3;
    param.coef0 = 0.1;
    param.kernel_type = SvmParam::POLY;
    kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        real cpu_kernel = powf(param.gamma * dot_product(dataSet.instances(), i, i) + param.coef0, param.degree);
        real gpu_kernel = kernelMatrix->diag()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = powf(param.gamma * dot_product(dataSet.instances(), rows[i], j) + param.coef0,
                                   param.degree);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << rows[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = powf(param.gamma * dot_product(dataSet.instances(), i, j) + param.coef0, param.degree);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i << "," << j;
        }
    }

    delete kernelMatrix;

    //test linear kernel
    param.kernel_type = SvmParam::LINEAR;
    kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        real cpu_kernel = dot_product(dataSet.instances(), i, i);
        real gpu_kernel = kernelMatrix->diag()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = dot_product(dataSet.instances(), rows[i], j);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << rows[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = dot_product(dataSet.instances(), i, j);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i << "," << j;
        }
    }

    delete kernelMatrix;

    //test sigmoid kernel
    param.gamma = 0.5;
    param.coef0 = 0.1;
    param.kernel_type = SvmParam::SIGMOID;
    kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        real cpu_kernel = tanhf(param.gamma * dot_product(dataSet.instances(), i, i) + param.coef0);
        real gpu_kernel = kernelMatrix->diag()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = tanhf(param.gamma * dot_product(dataSet.instances(), rows[i], j) + param.coef0);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << rows[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            real gpu_kernel = kernel_rows[i * kernelMatrix->n_instances() + j];
            real cpu_kernel = tanhf(param.gamma * dot_product(dataSet.instances(), i, j) + param.coef0);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-5) << i << "," << j;
        }
    }

    delete kernelMatrix;
}
