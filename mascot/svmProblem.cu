//
// Created by shijiashuai on 2016/11/1.
//

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "svmProblem.h"

void SvmProblem::groupClasses() {
    if (!subProblem) {
        vector<int> dataLabel(v_nLabels.size());

        //get the class labels; count the number of instances in each class.
        for (int i = 0; i < v_nLabels.size(); ++i) {
            int j;
            for (j = 0; j < label.size(); ++j) {
                if (v_nLabels[i] == label[j]) {
                    count[j]++;
                    break;
                }
            }
            dataLabel[i] = j;
            //if the label is unseen, add it to label vector.
            if (j == label.size()) {
                label.push_back(v_nLabels[i]);
                count.push_back(1);
            }
        }

        //logically put instances of the same class consecutively.
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
}

SvmProblem SvmProblem::getSubProblem(int i, int j) const {
    vector<vector<KeyValue> > v_vSamples;
    vector<int> v_nLabels;
    vector<int> originalIndex;
    vector<int> originalLabel;
    int si = start[i];
    int ci = count[i];
//    int ci = 5000;
    int sj = start[j];
    int cj = count[j];
//    int cj = 5000;
    for (int k = 0; k < ci; ++k) {
        v_vSamples.push_back(this->v_vSamples[perm[si + k]]);
        originalIndex.push_back(perm[si + k]);
        originalLabel.push_back(i);
        v_nLabels.push_back(+1);
    }
    for (int k = 0; k < cj; ++k) {
        v_vSamples.push_back(this->v_vSamples[perm[sj + k]]);
        originalIndex.push_back(perm[sj + k]);
        originalLabel.push_back(j);
        v_nLabels.push_back(-1);
    }
    SvmProblem subProblem(v_vSamples, numOfFeatures, v_nLabels);
    subProblem.label[0] = i;
    subProblem.label[1] = j;
//    subProblem.label.push_back(i);
//    subProblem.label.push_back(j);
//    subProblem.start.push_back(0);
//    subProblem.start.push_back(count[i]);
//    subProblem.count.push_back(count[i]);
//    subProblem.count.push_back(count[j]);
    subProblem.originalIndex = originalIndex;
    subProblem.originalLabel = originalLabel;
    subProblem.subProblem = true;
    return subProblem;
}

unsigned int SvmProblem::getNumOfClasses() const {
    return (unsigned int) label.size();
}

unsigned int SvmProblem::getNumOfSamples() const {
    return v_vSamples.size();
}

int SvmProblem::getNumOfFeatures() const {
    return numOfFeatures;
}

vector<vector<KeyValue> > SvmProblem::getOneClassSamples(int i) const {
    vector<vector<KeyValue> > samples;
    int si = start[i];
    int ci = count[i];
    for (int k = 0; k < ci; ++k) {
        samples.push_back(v_vSamples[perm[si + k]]);
    }
    return samples;
}

bool SvmProblem::isBinary() const {
    return 2 == this->getNumOfClasses();
}

