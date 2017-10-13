//
// Created by jiashuai on 17-10-6.
//

#include <iostream>
#include <iomanip>
#include <cstring>
#include <thundersvm/model/oneclass_svc.h>
using namespace std;

void OneClassSVC::train(DataSet dataset, SvmParam param) {
    int n_instances = dataset.total_count();
    SyncData<real> alpha(n_instances);
    SyncData<real> f_val(n_instances);

    KernelMatrix kernelMatrix(dataset.instances(), param.gamma);

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
    SyncData<int> idx(1);
    SyncData<real> kernel_row(n_instances);
    f_val.mem_set(0);
    for (int i = 0; i <= n; ++i) {
        idx[0] = i;
        kernelMatrix.get_rows(idx, kernel_row);
        for (int j = 0; j < n_instances; ++j) {
            f_val[j] += alpha[i] * kernel_row[j];
        }
    }

    SyncData<int> y(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        y[i] = 1;
    }
    smo_solver(kernelMatrix, y, alpha, rho, f_val, param.epsilon, 1, ws_size);

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
    ofstream libmod;
    string str = path + ".model";
    libmod.open(str.c_str());
    if (!libmod.is_open()) {
        cout << "can't open file " << path << endl;
        //return ret;
    }
    const SvmParam &param = this->param;
    const char *sType[] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "NULL"};  /* svm_type */
    const char *kType[] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};
    libmod << "svm_type " << sType[param.svm_type] << endl;
    libmod << "kernel_type " << kType[param.kernel_type] << endl;
    if (param.kernel_type == 1)
        libmod << "degree " << param.degree << endl;
    if (param.kernel_type == 1 || param.kernel_type == 2 || param.kernel_type == 3)/*1:poly 2:rbf 3:sigmoid*/
        libmod << "gamma " << param.gamma << endl;
    if (param.kernel_type == 1 || param.kernel_type == 3)
        libmod << "coef0 " << param.coef0 << endl;
    //unsigned int nr_class = this->dataSet.n_classes();
    unsigned int total_sv = sv.size();
    //libmod << "nr_class " << nr_class << endl;
    libmod << "total_sv " << total_sv << endl;
    libmod << "rho";
    libmod << " " << rho;
    libmod << endl;
    libmod << "SV" << endl;
    vector<real> sv_coef = this->coef;
    vector<vector<DataSet::node>> SV = this->sv;
    for(int i=0;i<total_sv;i++)
    {
        libmod << setprecision(16) << sv_coef[i]<< " ";

        vector<DataSet::node> p = SV[sv_index[i]];
        int k = 0;
        if(param.kernel_type == PRECOMPUTED)
            libmod << "0:" << p[k].value << " ";
        else
            for(; k < p.size(); k++)
            {
                libmod << p[k].index << ":" << setprecision(8) << p[k].value << " ";
            }
        libmod << endl;
    }
    libmod.close();
}

void OneClassSVC::load_from_file(string path) {
    int total_sv;
    real ftemp;
    //unsigned int cnr2 = 0;
    const char *sType[] = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr", "NULL"};  
    const char *kType[] = {"linear", "polynomial", "rbf", "sigmoid", "precomputed", "NULL"};

    ifstream ifs;
    path = path + ".model";
    ifs.open(path.c_str());//"dataset/a6a.model");
    if (!ifs.is_open())
        cout << "can't open file" << endl;
    string feature;
    while (ifs >> feature) {
        //cout<<"247"<<endl;
        // cout<<feature<<endl;
        //cout<<"feature"<<feature<<endl;
        if (feature == "svm_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (value == sType[i])
                    param.svm_type = i;
            }
        } else if (feature == "kernel_type") {
            string value;
            ifs >> value;
            for (int i = 0; i < 6; i++) {
                if (feature == kType[i])
                    param.kernel_type = i;
            }
        } else if (feature == "degree") {
            ifs >> param.degree;
        } else if (feature == "coef0") {
            ifs >> param.coef0;
        } else if (feature == "gamma") {
            ifs >> param.gamma;

        }/* else if (feature == "nr_class") {
            //ifs >> n_classes;
            //n_binary_models = n_classes * (n_classes - 1) / 2;
            //cnr2 = n_binary_models;
            //cout<<"cnr2:"<<cnr2<<endl;
            //ifs>>n_classes;
        } */else if (feature == "total_sv") {
            ifs >> total_sv;
            //total_sv = n_instances;
        } else if (feature == "rho") {
            /*
            vector<real> frho(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> frho[i];
            */
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
                    //coefT.push_back(ftemp);
                //cout<<"ftemp:"<<ftemp<<endl;
                //}
                //vector<DataSet::node> p = svT[i];
                getline(ifs, value);
                //cout<<"value:"<<value<<endl;
                sstr << value;
                //cout<<"sstr:"<<sstr<<endl;
                string temp;
                stringstream stemp;
                int indext;
                float valuet;
                //int k = 0;
                while (sstr >> temp) {
                    //cout<<"in sstr i :"<<i<<endl;
                    //if(i == 1)
                        //cout<<"in sstr"<<endl;
                    int ind = temp.find_first_of(":");
                    stemp << temp.substr(0, ind);
                    stemp >> indext;
                    //nodet.index = indext;
                    stemp.clear();
                    stemp << temp.substr(ind + 1, value.size());
                    stemp>>valuet;

                    //nodet.value = valuet;
                    DataSet::node nodet(indext, valuet);
                    svT[i].push_back(nodet);
                    stemp.clear();
                    sstr.clear();
                }
                //cout<<"p[0].index"<<
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
    //cout<<"sv_index[1]:"<<sv_index[1]<<endl;
    ifs.close();
}


