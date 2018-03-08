//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"

float_type rbf_kernel(const DataSet::node2d &instances, int x, int y, float_type gamma) {
    float_type sum = 0;
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

float_type dot_product(const DataSet::node2d &instances, int x, int y) {
    float_type sum = 0;
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
    unsigned n_rows = 10;
    SvmParam param;
    SyncArray<int> rows(n_rows);
    SyncArray<float_type> kernel_rows(n_rows * dataSet.n_instances());
    for (unsigned i = 0; i < n_rows; ++i) {
        rows.host_data()[i] = i * 3 + 4;
    }

    //test rbf kernel
    param.gamma = 0.5;
    param.kernel_type = SvmParam::RBF;
    KernelMatrix *kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    DataSet::node2d instances(dataSet.instances().begin(), dataSet.instances().begin() + n_rows);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        float_type cpu_kernel = rbf_kernel(dataSet.instances(), i, i, param.gamma);
        float_type gpu_kernel = kernelMatrix->diag().host_data()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = rbf_kernel(dataSet.instances(), rows.host_data()[i], j, param.gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << rows.host_data()[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = rbf_kernel(dataSet.instances(), i, j, param.gamma);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i << "," << j;
        }
    }
    delete kernelMatrix;

    //test poly kernel
    param.gamma = 0.5;
    param.degree = 3;
    param.coef0 = 0.1;
    param.kernel_type = SvmParam::POLY;
    kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        float_type cpu_kernel = powf(param.gamma * dot_product(dataSet.instances(), i, i) + param.coef0, param.degree);
        float_type gpu_kernel = kernelMatrix->diag().host_data()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = powf(
                    param.gamma * dot_product(dataSet.instances(), rows.host_data()[i], j) + param.coef0,
                    param.degree);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << rows.host_data()[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = powf(param.gamma * dot_product(dataSet.instances(), i, j) + param.coef0,
                                         param.degree);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i << "," << j;
        }
    }

    delete kernelMatrix;

    //test linear kernel
    param.kernel_type = SvmParam::LINEAR;
    kernelMatrix = new KernelMatrix(dataSet.instances(), param);
    kernelMatrix->get_rows(rows, kernel_rows);

    for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
        float_type cpu_kernel = dot_product(dataSet.instances(), i, i);
        float_type gpu_kernel = kernelMatrix->diag().host_data()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = dot_product(dataSet.instances(), rows.host_data()[i], j);
            EXPECT_NEAR(gpu_kernel, cpu_kernel,
                        1e-4) << rows.host_data()[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = dot_product(dataSet.instances(), i, j);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i << "," << j;
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
        float_type cpu_kernel = tanhf(param.gamma * dot_product(dataSet.instances(), i, i) + param.coef0);
        float_type gpu_kernel = kernelMatrix->diag().host_data()[i];
        EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i;
    }
    for (unsigned i = 0; i < rows.size(); ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = tanhf(
                    param.gamma * dot_product(dataSet.instances(), rows.host_data()[i], j) + param.coef0);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << rows.host_data()[i] << "," << j;
        }
    }

    kernelMatrix->get_rows(instances, kernel_rows);

    for (unsigned i = 0; i < n_rows; ++i) {
        for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
            float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
            float_type cpu_kernel = tanhf(param.gamma * dot_product(dataSet.instances(), i, j) + param.coef0);
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i << "," << j;
        }
    }

    delete kernelMatrix;
}
