/*
 * @author: Created by shijiashuai on 2016/11/1.
 */

#ifndef MASCOT_SVM_SVMPROBLEM_H
#define MASCOT_SVM_SVMPROBLEM_H

#include <vector>
#include"../svm-shared/gpu_global_utility.h"
using std::vector;
class SvmProblem {
public:
    vector<vector<svm_node> > v_vSamples;
    vector<int> v_nLabels;
    vector<int> count;
    vector<int> start;
    vector<int> perm;
    vector<int> label;
    SvmProblem(const vector<vector<svm_node> > &v_vSamples, const vector<int> &v_nLabels):
            v_vSamples(v_vSamples),v_nLabels(v_nLabels){
        this->groupClasses();
    }
    void groupClasses();
    SvmProblem getSubProblem(int i, int j) const;

    unsigned int getNumOfClasses() const;

    unsigned long long getNumOfSamples() const;

    unsigned long getNumOfFeatures() const;
};


#endif //MASCOT_SVM_SVMPROBLEM_H
