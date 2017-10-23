//
// Created by jiashuai on 17-10-6.
//

#include <iostream>
#include <iomanip>
#include <thundersvm/model/oneclass_svc.h>
using namespace std;
__global__ void init_f_kernel(real *f_val, const real *alpha, const real *kernel_row, int n, int n_instances) {
    KERNEL_LOOP(i, n_instances) {
        for (int j = 0; j <= n; ++j) {
            f_val[i] += alpha[j] * kernel_row[j * n_instances + i];
        }
    }
}

void OneClassSVC::train(DataSet dataset, SvmParam param) {
    int n_instances = dataset.total_count();
    SyncData<real> alpha(n_instances);
    SyncData<real> f_val(n_instances);

    KernelMatrix kernelMatrix(dataset.instances(), param);

    alpha.mem_set(0);
    int n = static_cast<int>(param.nu * n_instances);
    for (int i = 0; i < n; ++i) {
        alpha[i] = 1;
    }
    if (n < n_instances)
        alpha[n] = param.nu * n_instances - n;
    int ws_size = min(min(max2power(n), max2power(n_instances - n)) * 2, 1024);

    //init_f
    //TODO batch, thrust
    SyncData<int> idx(n + 1);
    SyncData<real> kernel_row(n_instances * (n + 1));
    f_val.mem_set(0);
    for (int i = 0; i <= n; ++i) {
        idx[i] = i;
    }
    kernelMatrix.get_rows(idx, kernel_row);
    SAFE_KERNEL_LAUNCH(init_f_kernel, f_val.device_data(), alpha.device_data(), kernel_row.device_data(), n,
                       n_instances);

    SyncData<int> y(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        y[i] = 1;
    }
    smo_solver(kernelMatrix, y, alpha, rho, f_val, param.epsilon, 1, 4);

    record_model(alpha, y, dataset.instances(), param);
}

vector<real> OneClassSVC::predict(const DataSet::node2d &instances, int batch_size) {
    vector<real> dec_values = SvmModel::predict(instances, batch_size);
    vector<real> predict_y;
    for (int i = 0; i < dec_values.size(); ++i) {
        predict_y.push_back(dec_values[i] > 0 ? 1 : -1);
    }
    return predict_y;
}

void OneClassSVC::save_to_file(string path) {
    ofstream fs_model;
    string file_name = path + ".model";
    fs_model.open(file_name.c_str());
    CHECK(fs_model.is_open()) << "file " << file_name << " not found";
    const SvmParam &param = this->param;
    fs_model << "svm_type " << svm_type_name[param.svm_type] << endl;
    fs_model << "kernel_type " << kernel_type_name[param.kernel_type] << endl;
    if (param.kernel_type == 1)
        fs_model << "degree " << param.degree << endl;
    if (param.kernel_type == 1 || param.kernel_type == 2 || param.kernel_type == 3)/*1:poly 2:rbf 3:sigmoid*/
        fs_model << "gamma " << param.gamma << endl;
    if (param.kernel_type == 1 || param.kernel_type == 3)
        fs_model << "coef0 " << param.coef0 << endl;
    //unsigned int nr_class = this->dataSet.n_classes();
    unsigned int total_sv = sv.size();
    //libmod << "nr_class " << nr_class << endl;
    fs_model << "total_sv " << total_sv << endl;
    fs_model << "rho";
    fs_model << " " << rho;
    fs_model << endl;
    fs_model << "SV" << endl;
    vector<real> sv_coef = this->coef;
    vector<vector<DataSet::node>> SV = this->sv;
    for(int i=0;i<total_sv;i++)
    {
        fs_model << setprecision(16) << sv_coef[i]<< " ";

        vector<DataSet::node> p = SV[sv_index[i]];
        int k = 0;
        if (param.kernel_type == SvmParam::PRECOMPUTED)
            fs_model << "0:" << p[k].value << " ";
        else
            for(; k < p.size(); k++)
            {
                fs_model << p[k].index << ":" << setprecision(8) << p[k].value << " ";
            }
        fs_model << endl;
    }
    fs_model.close();
}

void OneClassSVC::load_from_file(string path) {
    int total_sv;
    real ftemp;

    ifstream ifs;
    string file_name = path + ".model";
    ifs.open(file_name.c_str());
    CHECK(ifs.is_open()) << "file " << file_name << " not found";
    string feature;
    while (ifs >> feature) {
        if (feature == "svm_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (value == svm_type_name[i])
                    param.svm_type = i;
            }
        } else if (feature == "kernel_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (feature == kernel_type_name[i])
                    param.kernel_type = i;
            }
        } else if (feature == "degree") {
            ifs >> param.degree;
        } else if (feature == "coef0") {
            ifs >> param.coef0;
        } else if (feature == "gamma") {
            ifs >> param.gamma;

        }else if (feature == "total_sv") {
            ifs >> total_sv;
            //total_sv = n_instances;
        } else if (feature == "rho") {
            real frho;
            ifs >> frho;
            rho = frho;
        }
        else if (feature == "SV") {
            cout<<"SV"<<endl;
            string value;
            
            vector<real> coefT(total_sv);
            vector<vector<DataSet::node>> svT(total_sv);
            //DataSet::node nodet;
            for(int i=0;i<total_sv;i++)
            {
                //for(int j=0;j<nr_class;j++){
                stringstream sstr;
                    ifs >> ftemp;
                coefT[i] = ftemp;
                getline(ifs, value);
                sstr << value;
                string temp;
                stringstream stemp;
                int indext;
                float valuet;
                while (sstr >> temp) {
                    int ind = temp.find_first_of(":");
                    stemp << temp.substr(0, ind);
                    stemp >> indext;
                    stemp.clear();
                    stemp << temp.substr(ind + 1, value.size());
                    stemp>>valuet;

                    DataSet::node nodet(indext, valuet);
                    svT[i].push_back(nodet);
                    stemp.clear();
                    sstr.clear();
                }
            }
            coef = coefT;         
            sv = svT;
        }//end else if

    }//end while
    //svmParam = paramT;
    vector<int> sv_indext(total_sv);
    for(int i = 0; i < total_sv; i++)
        sv_indext[i] = i;
    sv_index = sv_indext;
    ifs.close();
}


