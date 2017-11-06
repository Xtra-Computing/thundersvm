//
// Created by jiashuai on 17-9-21.
//

#include <thundersvm/kernel/smo_kernel.h>
#include <thrust/sort.h>
#include <thundersvm/model/svmmodel.h>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <iomanip>

using std::ofstream;
using std::endl;
using std::setprecision;
using std::ifstream;
using std::stringstream;

const char *SvmParam::kernel_type_name[6] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};
const char *SvmParam::svm_type_name[6] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr",
                                          "NULL"};  /* svm_type */

vector<real> SvmModel::predict(const DataSet::node2d &instances, int batch_size) {
    //TODO use thrust
    //prepare device data
    int n_sv = coef.size();
    SyncData<real> coef(n_sv);
    SyncData<int> sv_index(n_sv);
    SyncData<int> sv_start(1);
    SyncData<int> sv_count(1);
    SyncData<real> rho(1);

    sv_start[0] = 0;
    sv_count[0] = n_sv;
    rho[0] = this->rho;
    coef.copy_from(this->coef.data(), n_sv);
    sv_index.copy_from(this->sv_index.data(), n_sv);

    //compute kernel values
    KernelMatrix k_mat(sv, param);

    auto batch_start = instances.begin();
    auto batch_end = batch_start;
    vector<real> predict_y;
    while (batch_end != instances.end()) {
        while (batch_end != instances.end() && batch_end - batch_start < batch_size) batch_end++;
        DataSet::node2d batch_ins(batch_start, batch_end);
        SyncData<real> kernel_values(batch_ins.size() * sv.size());
        k_mat.get_rows(batch_ins, kernel_values);
        SyncData<real> dec_values(batch_ins.size());

        //sum kernel values and get decision values
        SAFE_KERNEL_LAUNCH(kernel_sum_kernel_values, kernel_values.device_data(), batch_ins.size(), sv.size(),
                           1, sv_index.device_data(), coef.device_data(), sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), dec_values.device_data());

        for (int i = 0; i < batch_ins.size(); ++i) {
            predict_y.push_back(dec_values[i]);
        }
        batch_start += batch_size;
    }
    return predict_y;
}

void SvmModel::record_model(const SyncData<real> &alpha, const SyncData<int> &y, const DataSet::node2d &instances,
                            const SvmParam param) {
    int n_sv = 0;
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha[i] != 0) {
            coef.push_back(alpha[i]);
            sv_index.push_back(sv.size());
            sv.push_back(instances[i]);
            n_sv++;
        }
    }
    this->param = param;
    LOG(INFO) << "RHO = " << rho;
    LOG(INFO) << "#SV = " << n_sv;
}

vector<real> SvmModel::cross_validation(DataSet dataset, SvmParam param, int n_fold) {
    dataset.group_classes(this->param.svm_type == SvmParam::C_SVC);//group classes only for classification

    vector<real> y_test_all;
    vector<real> y_predict_all;

    for (int k = 0; k < n_fold; ++k) {
        LOG(INFO) << n_fold << " fold cross-validation(" << k + 1 << "/" << n_fold << ")";
        DataSet::node2d x_train, x_test;
        vector<real> y_train, y_test;
        for (int i = 0; i < dataset.n_classes(); ++i) {
            int fold_test_count = dataset.count()[i] / n_fold;
            vector<int> class_idx = dataset.original_index(i);
            auto idx_begin = class_idx.begin() + fold_test_count * k;
            auto idx_end = idx_begin;
            while (idx_end != class_idx.end() && idx_end - idx_begin < fold_test_count) idx_end++;
            for (int j: vector<int>(idx_begin, idx_end)) {
                x_test.push_back(dataset.instances()[j]);
                y_test.push_back(dataset.y()[j]);
            }
            class_idx.erase(idx_begin, idx_end);
            for (int j:class_idx) {
                x_train.push_back(dataset.instances()[j]);
                y_train.push_back(dataset.y()[j]);
            }
        }
        DataSet train_dataset(x_train, dataset.n_features(), y_train);
        this->train(train_dataset, param);
        vector<real> y_predict = this->predict(x_test, 1000);
        y_test_all.insert(y_test_all.end(), y_test.begin(), y_test.end());
        y_predict_all.insert(y_predict_all.end(), y_predict.begin(), y_predict.end());
    }
	vector<real> test_predict=y_test_all;
	test_predict.insert(test_predict.end(), y_predict_all.begin(), y_predict_all.end());
    return test_predict; 
}

void SvmModel::save_to_file(string path) {
    ofstream fs_model;
    fs_model.open(path.c_str());
    CHECK(fs_model.is_open()) << "create file " << path << "failed";
    const SvmParam &param = this->param;
    fs_model << "svm_type " << SvmParam::svm_type_name[param.svm_type] << endl;
    fs_model << "kernel_type " << SvmParam::kernel_type_name[param.kernel_type] << endl;
    if (param.kernel_type == SvmParam::POLY)
        fs_model << "degree " << param.degree << endl;
    if (param.kernel_type == SvmParam::POLY
        || param.kernel_type == SvmParam::RBF
        || param.kernel_type == SvmParam::SIGMOID)
        fs_model << "gamma " << param.gamma << endl;
    if (param.kernel_type == SvmParam::POLY || param.kernel_type == SvmParam::SIGMOID)
        fs_model << "coef0 " << param.coef0 << endl;
    fs_model << "nr_class " << 2 << endl;
    fs_model << "total_sv " << sv.size() << endl;
    fs_model << "rho " << rho << endl;
    fs_model << "SV " << endl;
    for (int i = 0; i < sv.size(); i++) {
        fs_model << setprecision(16) << coef[i] << " ";

        vector<DataSet::node> p = sv[sv_index[i]];
        int k = 0;
//        if (param.kernel_type == SvmParam::PRECOMPUTED)
//            fs_model << "0:" << p[k].value << " ";
//        else
        for (; k < p.size(); k++) {
            fs_model << p[k].index << ":" << setprecision(8) << p[k].value << " ";
        }
        fs_model << endl;
    }
    fs_model.close();
}

void SvmModel::load_from_file(string path) {
    int total_sv;
    ifstream ifs;
    ifs.open(path.c_str());
    CHECK(ifs.is_open()) << "file " << path << " not found";
    string feature;
    while (ifs >> feature) {
        if (feature == "svm_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (value == SvmParam::svm_type_name[i])
                    param.svm_type = static_cast<SvmParam::SVM_TYPE>(i);
            }
        } else if (feature == "kernel_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (feature == SvmParam::kernel_type_name[i])
                    param.kernel_type = static_cast<SvmParam::KERNEL_TYPE>(i);
            }
        } else if (feature == "degree") {
            ifs >> param.degree;
        } else if (feature == "coef0") {
            ifs >> param.coef0;
        } else if (feature == "gamma") {
            ifs >> param.gamma;

        } else if (feature == "total_sv") {
            ifs >> total_sv;
        } else if (feature == "rho") {
            ifs >> rho;
        } else if (feature == "SV") {
            string line;
            getline(ifs, line);
            for (int i = 0; i < total_sv; i++) {
                getline(ifs, line);
                stringstream ss(line);
                coef.emplace_back();
                ss >> coef.back();
                sv.emplace_back();//reserve space for an instance
                string tuple;
                while (ss >> tuple) {
                    sv.back().emplace_back(0, 0);
                    CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &sv.back().back().index, &sv.back().back().value), 2)
                        << "error when loading model file";
                };
            }//end else if
            sv_index.clear();
            for (int i = 0; i < total_sv; i++)
                sv_index.push_back(i);
            ifs.close();
        }
    }
}

