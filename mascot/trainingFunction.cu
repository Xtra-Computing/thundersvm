/*
 * testTrainer.cpp
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi Wen
 */

#include "trainingFunction.h"
#include <sys/time.h>
#include "DataIOOps/DataIO.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../SharedUtility/KeyValue.h"

extern float calculateKernelTime;
extern float preComputeTime;
extern float selectTime;
extern float updateAlphaTime;
extern float updateGValueTime;
extern float iterationTime;
void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel);

void trainSVM(SVMParam &param, string strTrainingFileName, int nNumofFeature, SvmModel &model, bool evaluteTrainingError) {
    timeval start, end;
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFileSparse(strTrainingFileName, nNumofFeature, v_v_Instance, v_nLabel);
    SvmProblem problem(v_v_Instance, nNumofFeature, v_nLabel);
    gettimeofday(&start,NULL);
    model.fit(problem, param);
    gettimeofday(&end,NULL);
    printf("training time: %fs\n", timeElapse(start, end));
    printf("iteration time : %fs\n", iterationTime);
    printf("kernel pre-computation time: %fs\n", preComputeTime);
    printf("2 instances selection time: %fs\n", selectTime);
    printf("kernel calculation time: %fs\n", calculateKernelTime);
    printf("alpha updating time: %fs\n",updateAlphaTime);
    printf("g value updating time: %fs\n",updateGValueTime);
    //evaluate training error
    if(evaluteTrainingError == true){
    	printf("Computing training accuracy...\n");
    	evaluate(model, v_v_Instance, v_nLabel);
    }
}

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int nNumofFeature) {
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    CDataIOOps rawDataRead;
    int nNumofInstance = 0;     //not used
    long long nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, nNumofFeature, nNumofInstance, nNumofValue);
    rawDataRead.ReadFromFileSparse(strTrainingFileName, nNumofFeature, v_v_Instance, v_nLabel);

    //evaluate testing error
    evaluate(model, v_v_Instance, v_nLabel);
}

/**
 * @brief: evaluate the svm model, given some labeled instances.
 */
void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel)
{
    //perform svm classification

    int batchSize = 2000;
    int begin = 0;
    vector<int> predictLabels;
    clock_t start, end;
    start = clock();
    while (begin < v_v_Instance.size()) {
        vector<vector<KeyValue> > samples(v_v_Instance.begin() + begin,
                                          v_v_Instance.begin() + min(begin + batchSize, (int) v_v_Instance.size()));
        vector<int> predictLabelPart = model.predict(samples, model.isProbability());
        predictLabels.insert(predictLabels.end(), predictLabelPart.begin(), predictLabelPart.end());
        begin += batchSize;
    }
    end = clock();
    int numOfCorrect = 0;
    for (int i = 0; i < v_v_Instance.size(); ++i) {
        if (predictLabels[i] == v_nLabel[i])
            numOfCorrect++;
    }
    printf("classifier accuracy = %.2f%%(%d/%d)\n", numOfCorrect / (float) v_v_Instance.size() * 100,
           numOfCorrect, (int) v_v_Instance.size());
    printf("prediction time elapsed: %.2fs\n", (float) (end - start) / CLOCKS_PER_SEC);
}
