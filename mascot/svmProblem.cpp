//
// Created by shijiashuai on 2016/11/1.
//

#include "svmProblem.h"

void svmProblem::groupClasses() {
    vector<int> dataLabel(v_nLabels.size());
    for (int i = 0; i < v_nLabels.size(); ++i) {
        int j;
        for (j = 0; j < label.size(); ++j) {
            if (v_nLabels[i] == label[j]) {
                count[j]++;
                break;
            }
        }
        dataLabel[i] = j;
        //if the label is unseen, add it to label set
        if (j == label.size()) {
            label.push_back(v_nLabels[i]);
            count.push_back(1);
        }
    }

// Labels are ordered by their first occurrence in the training set.
// However, for two-class sets with -1/+1 labels and -1 appears first,
// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.

    if (label.size() == 2 && label[0] == -1 && label[1] == 1) {
        label[0] = 1;
        label[1] = -1;
        int t = count[0];
        count[0] = count[1];
        count[1] = t;
        for (int i = 0; i < dataLabel.size(); i++) {
            if (dataLabel[i] == 0)
                dataLabel[i] = 1;
            else
                dataLabel[i] = 0;
        }
    }

    start.push_back(0);
    for (int i = 1; i < count.size(); ++i) {
        start.push_back(start[i - 1] + count[i - 1]);
    }
    vector<int> _start(start);
    perm = vector<int>(v_nLabels.size());
    for (int i = 0; i < v_nLabels.size(); ++i) {
        perm[_start[dataLabel[i]]] = i;
        _start[dataLabel[i]]++;
    }
}

svmProblem svmProblem::getSubProblem(int i, int j) const {
    vector<vector<float_point> > v_vSamples;
    vector<int> v_nLabels;
    int si = start[i];
    int ci = count[i];
    int sj = start[j];
    int cj = count[j];
    for (int k = 0; k < ci; ++k) {
        v_vSamples.push_back(this->v_vSamples[perm[si + k]]);
        v_nLabels.push_back(+1);
    }
    for (int k = 0; k < cj; ++k) {
        v_vSamples.push_back(this->v_vSamples[perm[sj + k]]);
        v_nLabels.push_back(-1);
    }
    return svmProblem(v_vSamples, v_nLabels);
}

unsigned int svmProblem::getNumOfClasses() const {
    return (unsigned int) label.size();
}

unsigned long long svmProblem::getNumOfSamples() const {
    return v_vSamples.size();
}

unsigned long svmProblem::getNumOfFeatures() const {
    return v_vSamples.front().size();
}
