//
// Created by jiashuai on 17-9-20.
//
#include "gtest/gtest.h"
#include "thundersvm/kernelmatrix.h"


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

kernel_type get_cpu_kernel(const DataSet::node2d &instances, int x, int y, const SvmParam &param) {
    kernel_type r;
    switch (param.kernel_type) {
        case SvmParam::RBF: {
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
            r = expf(-param.gamma * sum);
            break;
        }
        case SvmParam::POLY:
            r = powf(param.gamma * dot_product(instances, x, y) + param.coef0, param.degree);
            break;
        case SvmParam::LINEAR:
            r = dot_product(instances, x, y);
            break;
        case SvmParam::SIGMOID:
            r = tanhf(param.gamma * dot_product(instances, x, y) + param.coef0);
    }
    return r;
}

class KernelMatrixTest : public ::testing::Test {
    DataSet dataSet;
    unsigned n_rows = 10;
    SyncArray<int> rows;
    DataSet::node2d instances;
    SyncArray<kernel_type> kernel_rows;
protected:
    SvmParam param;

    void SetUp() override {
        param = SvmParam();
        dataSet.load_from_file(DATASET_DIR "test_dataset.txt");
        rows.resize(n_rows);
        kernel_rows.resize(n_rows * dataSet.n_instances());
        for (unsigned i = 0; i < n_rows; ++i) {
            rows.host_data()[i] = i * 3 + 4;
        }
        instances = DataSet::node2d(dataSet.instances().begin(), dataSet.instances().begin() + n_rows);
    }

    void TearDown() override {
        KernelMatrix *kernelMatrix = new KernelMatrix(dataSet.instances(), param);
        kernelMatrix->get_rows(rows, kernel_rows);

        //test diagonal elements
        for (unsigned i = 0; i < kernelMatrix->diag().size(); ++i) {
            float_type cpu_kernel = get_cpu_kernel(dataSet.instances(), i, i, param);
            float_type gpu_kernel = kernelMatrix->diag().host_data()[i];
            EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i;
        }

        //test get rows with index
        for (unsigned i = 0; i < rows.size(); ++i) {
            for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
                float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
                float_type cpu_kernel = get_cpu_kernel(dataSet.instances(), rows.host_data()[i], j, param);
                EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << rows.host_data()[i] << "," << j;
            }
        }

        //test get rows with instances
        kernelMatrix->get_rows(instances, kernel_rows);

        for (unsigned i = 0; i < n_rows; ++i) {
            for (unsigned j = 0; j < kernelMatrix->n_instances(); ++j) {
                float_type gpu_kernel = kernel_rows.host_data()[i * kernelMatrix->n_instances() + j];
                float_type cpu_kernel = get_cpu_kernel(dataSet.instances(), i, j, param);
                EXPECT_NEAR(gpu_kernel, cpu_kernel, 1e-4) << i << "," << j;
            }
        }
        delete kernelMatrix;
    }
};

TEST_F(KernelMatrixTest, test_rbf) {
    param.gamma = 0.5;
    param.kernel_type = SvmParam::RBF;
}

TEST_F(KernelMatrixTest, test_poly) {
    param.gamma = 0.5;
    param.degree = 3;
    param.coef0 = 0.1;
    param.kernel_type = SvmParam::POLY;
}

TEST_F(KernelMatrixTest, test_linear) {
    param.kernel_type = SvmParam::LINEAR;
}

TEST_F(KernelMatrixTest, test_sigmoid) {
    param.gamma = 0.5;
    param.coef0 = 0.1;
    param.kernel_type = SvmParam::SIGMOID;
}

