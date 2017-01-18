/*
 * @author: Created by shijiashuai on 2016/11/1.
 */

#ifndef MASCOT_SVM_SVMPROBLEM_H
#define MASCOT_SVM_SVMPROBLEM_H

#include <vector>
#include <cusparse.h>
#include "../SharedUtility/KeyValue.h"

using std::vector;

class SvmProblem {
public:
    vector<vector<KeyValue> > v_vSamples;
    vector<int> v_nLabels;
    vector<int> count;
    vector<int> start;
    vector<int> perm;						//put instances of the same class continuously
    vector<int> label;
    int numOfFeatures;
    bool subProblem;

    int getNumOfFeatures() const;

    //for subProblem
    vector<int> originalIndex;
    vector<int> originalLabel;

    SvmProblem(const vector<vector<KeyValue> > &v_vSamples, int numOfFeatures, const vector<int> &v_nLabels) :
            v_vSamples(v_vSamples), v_nLabels(v_nLabels), numOfFeatures(numOfFeatures), subProblem(false){
        this->groupClasses();
    }

    void groupClasses();

    SvmProblem getSubProblem(int i, int j) const;
    vector<vector<KeyValue> > getOneClassSamples(int i) const;

    unsigned int getNumOfClasses() const;

    unsigned int getNumOfSamples() const;
    bool isBinary() const;
};


#endif //MASCOT_SVM_SVMPROBLEM_H
