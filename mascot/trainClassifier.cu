/*
 * testTrainer.cpp
 *
 *  Created on: 31/10/2013
 *      Author: Zeyi Wen
 */

#include <sys/time.h>
#include "multiPredictor.h"
#include "trainClassifier.h"
#include "SVMCmdLineParser.h"
#include "classifierEvaluater.h"
#include "../svm-shared/Cache/cache.h"
#include "../svm-shared/HessianIO/deviceHessianOnFly.h"
#include "../SharedUtility/Timer.h"
#include "../SharedUtility/KeyValue.h"
#include "../SharedUtility/DataReader/LibsvmReaderSparse.h"

void trainSVM(SVMParam &param, string strTrainingFileName, int numFeature, SvmModel &model, bool evaluteTrainingError) {
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    int numInstance = 0;     //not used
    uint nNumofValue = 0;  //not used
    if(SVMCmdLineParser::numFeature > 0){
    	numFeature = SVMCmdLineParser::numFeature;
    }
    else
    	BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
	LibSVMDataReader drHelper;
	drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);
    SvmProblem problem(v_v_Instance, numFeature, v_nLabel);
    printf("yes\n");
//    problem = problem.getSubProblem(0,1);
    model.fit(problem, param);
    PRINT_TIME("training", trainingTimer)
    PRINT_TIME("working set selection",selectTimer)
    PRINT_TIME("pre-computation kernel",preComputeTimer)
    PRINT_TIME("iteration",iterationTimer)
    PRINT_TIME("g value updating",updateGTimer)
	model.saveLibModel(strTrainingFileName,problem);//save model in the same format as LIBSVM's
//    PRINT_TIME("2 instances selection",selectTimer)
//    PRINT_TIME("kernel calculation",calculateKernelTimer)
//    PRINT_TIME("alpha updating",updateAlphaTimer)
//    PRINT_TIME("init cache",initTimer)
    //evaluate training error
    if (evaluteTrainingError == true) {
        printf("Computing training accuracy...\n");
        evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::trainingError);
	}
}

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int numFeature) {
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    int numInstance = 0;     //not used
    uint nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
	LibSVMDataReader drHelper;
	drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);

    //evaluate testing error
    evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::testingError);
}

/**
 * @brief: evaluate the svm model, given some labeled instances.
 */
void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
			  vector<real> &classificationError){
    int batchSize = 1000;

    //create a miss labeling matrix for measuring the sub-classifier errors.
    model.missLabellingMatrix = vector<vector<int> >(model.nrClass, vector<int>(model.nrClass, 0));
    bool bEvaluateSubClass = true; //choose whether to evaluate sub-classifiers
    if(model.nrClass == 2)  //absolutely not necessary to evaluate sub-classifers
        bEvaluateSubClass = false;

    MultiPredictor predictor(model, model.param);

    int begin = 0;
    vector<int> predictLabels;
    TIMER_START(predictionTimer)
    while (begin < v_v_Instance.size()) {
    	//get a subset of instances
    	int end = min(begin + batchSize, (int) v_v_Instance.size());
        vector<vector<KeyValue> > samples(v_v_Instance.begin() + begin,
                                          v_v_Instance.begin() + end);
        vector<int> vLabel(v_nLabel.begin() + begin, v_nLabel.begin() + end);
        if(bEvaluateSubClass == false)
        	vLabel.clear();
        //predict labels for the subset of instances
        vector<int> predictLabelPart = predictor.predict(samples, vLabel);
        predictLabels.insert(predictLabels.end(), predictLabelPart.begin(), predictLabelPart.end());
        begin += batchSize;
    }
    int numOfCorrect = 0;
    for (int i = 0; i < v_v_Instance.size(); ++i) {
        if (predictLabels[i] == v_nLabel[i])
            numOfCorrect++;
    }
    TIMER_STOP(predictionTimer)
    printf("classifier accuracy = %.2f%%(%d/%d)\n", numOfCorrect / (float) v_v_Instance.size() * 100,
           numOfCorrect, (int) v_v_Instance.size());
    PRINT_TIME("prediction time elapsed:", predictionTimer)
    if(bEvaluateSubClass == true){
    	ClassifierEvaluater::evaluateSubClassifier(model.missLabellingMatrix, classificationError);
    }
}
