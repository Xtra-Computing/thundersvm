//
// Created by jiashuai on 17-10-5.
//
#include <iostream>
#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <thundersvm/solver/csmosolver.h>
#include "thundersvm/model/svr.h"
void SVR::train(DataSet dataset, SvmParam param) {
    int n_instances = dataset.total_count();

    //duplicate instances
    DataSet::node2d instances_2(dataset.instances());
    instances_2.insert(instances_2.end(), dataset.instances().begin(), dataset.instances().end());

    KernelMatrix kernelMatrix(instances_2, param);

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
    CSMOSolver solver;
    solver.solve(kernelMatrix, y, alpha_2, rho, f_val, param.epsilon, param.C, param.C, ws_size);
    SyncData<real> alpha(n_instances);
    for (int i = 0; i < n_instances; ++i) {
        alpha[i] = alpha_2[i] - alpha_2[i + n_instances];
    }
    record_model(alpha, y, dataset.instances(), param);
}

//void SVR::load_from_file(string path) {
//    //SvmParam paramT;
//    int nr_class = 0;
//    int total_sv;
//    real ftemp;
//
//    ifstream ifs;
//    string file_name = path + ".model";
//    ifs.open(file_name.c_str());
//    CHECK(ifs.is_open()) << "file " << file_name << " not found";
//    string feature;
//    while (ifs >> feature) {
//        if (feature == "svm_type") {
//            string value;
//            ifs >> value;
//            for (int i = 0; i < 6; i++) {
//                if (value == svm_type_name[i])
//                    param.svm_type = i;
//            }
//        } else if (feature == "kernel_type") {
//            string value;
//            ifs >> value;
//            for (int i = 0; i < 6; i++) {
//                if (feature == kernel_type_name[i])
//                    param.kernel_type = i;
//            }
//        } else if (feature == "degree") {
//            ifs >> param.degree;
//        } else if (feature == "coef0") {
//            ifs >> param.coef0;
//        } else if (feature == "gamma") {
//            ifs >> param.gamma;
//
//        }
//        else if (feature == "total_sv") {
//            ifs >> total_sv;
//            //total_sv = n_instances;
//        } else if (feature == "rho") {
//            real frho;
//            ifs >> frho;
//            rho = frho;
//        }
//        else if (feature == "nr_sv") {
//            ifs >> nr_class;
//        }
//        else if (feature == "SV") {
//            string value;
//
//            vector<real> coefT(total_sv);
//            vector<vector<DataSet::node>> svT(total_sv);
//            //DataSet::node nodet;
//            for(int i=0;i<total_sv;i++)
//            {
//                stringstream sstr;
//                ifs >> ftemp;
//                coefT[i] = ftemp;
//                getline(ifs, value);
//                sstr << value;
//                string temp;
//                stringstream stemp;
//                int indext;
//                float valuet;
//                //int k = 0;
//                while (sstr >> temp) {
//                    int ind = temp.find_first_of(":");
//                    stemp << temp.substr(0, ind);
//                    stemp >> indext;
//                    //nodet.index = indext;
//                    stemp.clear();
//                    stemp << temp.substr(ind + 1, value.size());
//                    stemp>>valuet;
//
//                    //nodet.value = valuet;
//                    DataSet::node nodet(indext, valuet);
//                    svT[i].push_back(nodet);
//                    stemp.clear();
//                    sstr.clear();
//                }
//            }
//            coef = coefT;
//            sv = svT;
//        }//end else if
//
//    }//end while
//    vector<int> sv_indext(total_sv);
//    for(int i = 0; i < total_sv; i++)
//        sv_indext[i] = i;
//    sv_index = sv_indext;
//    ifs.close();
//}

