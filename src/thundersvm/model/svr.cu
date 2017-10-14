//
// Created by jiashuai on 17-10-5.
//
#include <iostream>
#include <iomanip>
#include <cstring>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include "thundersvm/model/svr.h"
using namespace std;
void SVR::train(DataSet dataset, SvmParam param) {
    int n_instances = dataset.total_count();

    //duplicate instances
    DataSet::node2d instances_2(dataset.instances());
    instances_2.insert(instances_2.end(), dataset.instances().begin(), dataset.instances().end());

    KernelMatrix kernelMatrix(instances_2, param.gamma);

    SyncData<real> f_val(n_instances * 2);
    SyncData<int> y(n_instances * 2);

    for (int i = 0; i < n_instances; ++i) {
        f_val[i] = param.p - dataset.y()[i];
        y[i] = +1;
        f_val[i + n_instances] = -param.p - dataset.y()[i];
        y[i + n_instances] = -1;
    }

    SyncData<real> alpha_2(n_instances * 2);
    alpha_2.mem_set(0);
    int ws_size = min(max2power(n_instances) * 2, 1024);
    smo_solver(kernelMatrix, y, alpha_2, rho, f_val, param.epsilon, param.C, ws_size);
    SyncData<real> alpha(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        alpha[i] = alpha_2[i] - alpha_2[i + n_instances];
    }
    record_model(alpha, y, dataset.instances(), param);
}


void SVR::save_to_file(string path) {
    //bool ret = false;
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
    //unsigned int total_sv = this->dataSet.total_count();            //not sure
    //unsigned int nr_class = n_classes;
    unsigned int total_sv = sv.size();
    //libmod << "nr_class " << nr_class << endl;
    libmod << "total_sv " << total_sv << endl;
    libmod << "rho";
    libmod << " " << rho;
    libmod << endl;
    libmod << "SV" << endl;
    vector<real> sv_coef = this->coef;
    vector<vector<DataSet::node>> SV = this->sv;
    //cout<<"201"<<endl;
    for(int i=0;i<total_sv;i++)
    {
        libmod << setprecision(16) << sv_coef[i]<< " ";

        vector<DataSet::node> p = SV[sv_index[i]];
        int k = 0;
        //cout<<"210"<<endl;
        if(param.kernel_type == PRECOMPUTED)
            //fprintf(fp,"0:%d ",(int)(p->value));
            libmod << "0:" << p[k].value << " ";
        else
            for(; k < p.size(); k++)
            {
                //fprintf(fp,"%d:%.8g ",p->index,p->value);
                //cout<<"218"<<endl;
                libmod << p[k].index << ":" << setprecision(8) << p[k].value << " ";
            }
        //cout<<"222"<<endl;
        //fprintf(fp, "\n");
        libmod << endl;
    }
    libmod.close();
}

void SVR::load_from_file(string path) {
    //SvmParam paramT;
    int nr_class = 0;
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

        } 
        /*else if (feature == "nr_class") {
            //ifs >> n_classes;
            //n_binary_models = n_classes * (n_classes - 1) / 2;
            //cnr2 = n_binary_models;
            //cout<<"cnr2:"<<cnr2<<endl;
            //ifs>>n_classes;
        }*/ else if (feature == "total_sv") {
            ifs >> total_sv;
            //total_sv = n_instances;
        } else if (feature == "rho") {
            real frho;
            ifs >> frho;
            rho = frho;
        }
        else if (feature == "nr_sv") {
            ifs >> nr_class;
        } 
        else if (feature == "SV") {
            string value;
            
            vector<real> coefT(total_sv);
            vector<vector<DataSet::node>> svT(total_sv);
            //DataSet::node nodet;
            for(int i=0;i<total_sv;i++)
            {
                stringstream sstr;
                ifs >> ftemp;
                coefT[i] = ftemp;
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
    ifs.close();
}




