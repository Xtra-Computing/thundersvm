//
// Created by jiashuai on 17-9-21.
//
#include <iostream>
#include <iomanip>
#include <thundersvm/kernel/smo_kernel.h>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <thundersvm/model/svc.h>
#include "thrust/sort.h"
using namespace std;

void SVC::train(DataSet dataset, SvmParam param) {
    dataset.group_classes();
    this->label = dataset.label();
    n_classes = dataset.n_classes();
    n_binary_models = n_classes * (n_classes - 1) / 2;
    rho.resize(n_binary_models);
    sv_index.resize(n_binary_models);
    coef.resize(n_binary_models);
    this->param = param;
    int k = 0;
    for (int i = 0; i < n_classes; ++i) {
        for (int j = i + 1; j < n_classes; ++j) {
            DataSet::node2d ins = dataset.instances(i, j);//get instances of class i and j
            SyncData<int> y(ins.size());
            SyncData<real> alpha(ins.size());
            SyncData<real> f_val(ins.size());
            real rho;
            alpha.mem_set(0);
            for (int l = 0; l < dataset.count()[i]; ++l) {
                y[l] = +1;
                f_val[l] = -1;
            }
            for (int l = 0; l < dataset.count()[j]; ++l) {
                y[dataset.count()[i] + l] = -1;
                f_val[dataset.count()[i] + l] = +1;
            }
            KernelMatrix k_mat(ins, param);
            int ws_size = min(min(max2power(dataset.count()[0]), max2power(dataset.count()[1])) * 2, 1024);
            smo_solver(k_mat, y, alpha, rho, f_val, param.epsilon, param.C, ws_size);
            record_binary_model(k, alpha, y, rho, dataset.original_index(i, j), dataset.instances());
            k++;
        }
    }
}

vector<real> SVC::predict(const DataSet::node2d &instances, int batch_size) {
    //prepare device data
    SyncData<int> sv_start(n_binary_models);
    SyncData<int> sv_count(n_binary_models);
    int n_sv = 0;
    for (int i = 0; i < n_binary_models; ++i) {
        sv_start[i] = n_sv;
        sv_count[i] = this->coef[i].size();
        n_sv += this->coef[i].size();
    }
    SyncData<real> coef(n_sv);
    SyncData<int> sv_index(n_sv);
    SyncData<real> rho(n_binary_models);
    for (int i = 0; i < n_binary_models; ++i) {
        CUDA_CHECK(cudaMemcpy(coef.device_data() + sv_start[i], this->coef[i].data(), sizeof(real) * sv_count[i],
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sv_index.device_data() + sv_start[i], this->sv_index[i].data(), sizeof(int) * sv_count[i],
                              cudaMemcpyHostToDevice));
    }
    rho.copy_from(this->rho.data(), rho.count());

    //compute kernel values
    KernelMatrix k_mat(sv, param);

    auto batch_start = instances.begin();
    auto batch_end = batch_start;
    vector<real> predict_y;
    while (batch_end != instances.end()) {
        while (batch_end != instances.end() && batch_end - batch_start < batch_size) batch_end++;

        DataSet::node2d batch_ins(batch_start, batch_end);//get a batch of instances
        SyncData<real> kernel_values(batch_ins.size() * sv.size());
        k_mat.get_rows(batch_ins, kernel_values);
        SyncData<real> dec_values(batch_ins.size() * n_binary_models);

        //sum kernel values and get decision values
        SAFE_KERNEL_LAUNCH(kernel_sum_kernel_values, kernel_values.device_data(), batch_ins.size(), sv.size(),
                           n_binary_models, sv_index.device_data(), coef.device_data(), sv_start.device_data(),
                           sv_count.device_data(), rho.device_data(), dec_values.device_data());

        //predict y by voting among k(k-1)/2 models
        for (int l = 0; l < batch_ins.size(); ++l) {
            vector<int> votes(n_binary_models, 0);
            int k = 0;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    if (dec_values[l * n_binary_models + k] > 0)
                        votes[i]++;
                    else
                        votes[j]++;
                    k++;
                }
            }
            int maxVoteClass = 0;
            for (int i = 0; i < n_classes; ++i) {
                if (votes[i] > votes[maxVoteClass])
                    maxVoteClass = i;
            }
            predict_y.push_back((float) this->label[maxVoteClass]);
        }
        batch_start += batch_size;
    }
    return predict_y;
}

void SVC::save_to_file(string path) {
    ofstream fs_model;
    string str = path + ".model";
    fs_model.open(str.c_str());
    CHECK(fs_model.is_open()) << "file " << path << " not found";
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
    //unsigned int total_sv = this->dataSet.total_count();            //not sure
    unsigned int nr_class = n_classes;
    unsigned int total_sv = sv.size();
    //unsigned int total_sv = n_instances;
    fs_model << "nr_class " << nr_class << endl;
    fs_model << "total_sv " << total_sv << endl;
    vector<real> frho = rho;
    fs_model << "rho";
    for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++) {
        fs_model << " " << frho[i];
    }
    fs_model << endl;
    
    if (param.svm_type == 0) {
        fs_model << "label";
        for (int i = 0; i < nr_class; i++)
            fs_model << " " << label[i];
        fs_model << endl;
    }
    
    //cout<<"149"<<endl;
    /*
    if (this->probability) // regression has probA only
    {
        libmod << "probA";
        for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
            libmod << " " << probA[i];
        libmod << endl;
        libmod << "probB";
        for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
            libmod << " " << probB[i];
        libmod << endl;
    }
    */
    /*
    if (param.svm_type == 0)//c-svm
    {
        libmod << "nr_sv";
        for (int i = 0; i < nr_class; i++)
            libmod << " " << this->dataSet.count()[i];
        libmod << endl;
    }
    */
    fs_model << "SV" << endl;
    
    vector<vector<real>> sv_coef = this->coef;
    vector<vector<DataSet::node>> SV = this->sv;
    for(int i=0;i<total_sv;i++)
    {
        for(int j = 0; j < n_binary_models; j++){
            fs_model << setprecision(16) << sv_coef[j][i]<< " ";

        }
        for(int j = 0; j < n_binary_models; j++){
            vector<DataSet::node> p = SV[sv_index[j][i]];
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
    }
    fs_model.close();

}

void SVC::load_from_file(string path) {
    //SvmParam paramT;
    //int nr_class = 0;
    /*
    int total_sv;
    float ftemp;
    unsigned int cnr2 = 0;
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

        } else if (feature == "nr_class") {
            ifs >> n_classes;
            n_binary_models = n_classes * (n_classes - 1) / 2;
            cnr2 = n_binary_models;
            //cout<<"cnr2:"<<cnr2<<endl;
        } 
        else if (feature == "total_sv") {
            ifs >> total_sv;
        } else if (feature == "rho") {
            vector<real> frho(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> frho[i];
            rho = frho;
        }

        
        else if (feature == "label") {
            vector<int> ilabel(n_classes, 0);
            for (int i = 0; i < n_classes; i++)
                ifs >> ilabel[i];
            dataSet.label_ = ilabel;
        } else if (feature == "probA") {
            vector<real> fprobA(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> fprobA[i];
            model.probA = fprobA;
        } else if (feature == "probB") {
            vector<real> fprobB(cnr2, 0);
            for (int i = 0; i < cnr2; i++)
                ifs >> fprobB[i];
            model.probB = fprobB;
        } 

        else if (feature == "nr_sv") {
            vector<int> fnSV(nr_class, 0);
            for (int i = 0; i < nr_class; i++)
                ifs >> fnSV[i];
            dataSet.count_ = fnSV;
        } 
        
        else if (feature == "SV") {
            //cout<<"309"<<endl;
            string value;
            stringstream sstr;
            vector<vector<real>> coefT(cnr2);
            vector<vector<DataSet::node>> svT(total_sv);
            for(int i=0;i<total_sv;i++)
            {
                //cout<<"316"<<endl;
                for(int j=0;j<n_classes;j++){
                    ifs >> ftemp;
                    coefT[j].push_back(ftemp);
                }
                //cout<<"325"<<endl;
                vector<DataSet::node> p = svT[i];
                //int k = 0;
                getline(ifs, value);
                sstr << value;
                string temp;
                stringstream stemp;
                //int k = 0;
                //cout<<"336"<<endl;
                int indext;
                float valuet;
                DataSet::node nodet(0,0);
                while (sstr >> temp) {
                    int ind = temp.find_first_of(":");
                    stemp << temp.substr(0, ind);
                    //cout<<"340"<<endl;
                    stemp >> indext;

                    nodet.index = indext;
                    stemp.clear();
                    stemp << temp.substr(ind + 1, value.size());
                    stemp>>valuet;
                    nodet.value = valuet;

                    p.push_back(nodet);
                    stemp.clear();
                    //k++;
                    //cout<<"347"<<endl;
                }
            }
            coef = coefT;
            sv = svT;
        }//end else if

    }//end while
    ifs.close();
    //param = paramT;
    */
}


void SVC::record_binary_model(int k, const SyncData<real> &alpha, const SyncData<int> &y, real rho,
                              const vector<int> &original_index, const DataSet::node2d &original_instance) {
    int n_sv = 0;
    for (int i = 0; i < alpha.count(); ++i) {
        if (alpha[i] != 0) {
            coef[k].push_back(alpha[i] * y[i]);
            if (sv_index_map.find(original_index[i]) == sv_index_map.end()) {
                int sv_index = sv_index_map.size();
                sv_index_map[original_index[i]] = sv_index;
                sv.push_back(original_instance[original_index[i]]);
            }
            sv_index[k].push_back(sv_index_map[original_index[i]]);//save unique sv id.
            n_sv++;
        }
    }
    this->rho[k] = rho;
    LOG(INFO) << "rho=" << rho;
    LOG(INFO) << "#SV=" << n_sv;
}


