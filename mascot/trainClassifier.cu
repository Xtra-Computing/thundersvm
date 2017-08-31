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
    //    evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::trainingError);//uncomment!!!!!!!!!!!!!!1
	}
}

void evaluateSVMClassifier(SvmModel &model, string strTrainingFileName, int numFeature) {
    cout<<" start decision value"<<endl;
    vector<vector<KeyValue> > v_v_Instance;
    vector<int> v_nLabel;

    int numInstance = 0;     //not used
    uint nNumofValue = 0;  //not used
    BaseLibSVMReader::GetDataInfo(strTrainingFileName, numFeature, numInstance, nNumofValue);
	LibSVMDataReader drHelper;
	drHelper.ReadLibSVMAsSparse(v_v_Instance, v_nLabel, strTrainingFileName, numFeature);
    //evaluate testing error
    //evaluate(model, v_v_Instance, v_nLabel, ClassifierEvaluater::testingError,ofs);//uncomment!!!!!!!!!!111
}

/**
 * @brief: evaluate the svm model, given some labeled instances.
 */
void evaluate(SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
			  vector<real> &classificationError, ofstream &ofs){
    int batchSize = 10000;

    //create a miss labeling matrix for measuring the sub-classifier errors.
    model.missLabellingMatrix = vector<vector<int> >(model.nrClass, vector<int>(model.nrClass, 0));
    bool bEvaluateSubClass = true; //choose whether to evaluate sub-classifiers
    if(model.nrClass == 2)  //absolutely not necessary to evaluate sub-classifers
        bEvaluateSubClass = false;

    MultiPredictor predictor(model, model.param);

	clock_t start, finish;
    start = clock();
    int begin = 0;
    vector<int> predictLabels;
    while (begin < v_v_Instance.size()) {
    	//get a subset of instances
    	int end = min(begin + batchSize, (int) v_v_Instance.size());
        vector<vector<KeyValue> > samples(v_v_Instance.begin() + begin,
                                          v_v_Instance.begin() + end);
        vector<int> vLabel(v_nLabel.begin() + begin, v_nLabel.begin() + end);
        if(bEvaluateSubClass == false)
        	vLabel.clear();
        //predict labels for the subset of instances
	cout<<"in evaluate**************88"<<endl;
        vector<int> predictLabelPart = predictor.predict(samples, vLabel);
        predictLabels.insert(predictLabels.end(), predictLabelPart.begin(), predictLabelPart.end());
        begin += batchSize;
    }
    finish = clock();
    int numOfCorrect = 0;
    int neg=0;
    for (int i = 0; i < v_v_Instance.size(); ++i) {
	if(predictLabels[i]==1){
	neg++;
	}
        if (predictLabels[i] == v_nLabel[i])
            numOfCorrect++;
    }
    printf("binary classifier accuracy = %.2f%%(%d/%d)\n", numOfCorrect / (float) v_v_Instance.size() * 100,
           numOfCorrect, (int) v_v_Instance.size());
	ofs<<"binary training accuracy: "<<numOfCorrect/(float) v_v_Instance.size()*100<<" ("<<numOfCorrect<<"/"<<v_v_Instance.size()<<")"<<"\n";
    printf("prediction time elapsed: %.2fs\n", (float) (finish - start) / CLOCKS_PER_SEC);

    if(bEvaluateSubClass == true){
    	ClassifierEvaluater::evaluateSubClassifier(model.missLabellingMatrix, classificationError);
    }
}

float evaluateOVABinaryClassifier(vector<real>  &combDecValue, vector<vector<int> > &combPredictLabels, SvmModel &model, vector<vector<KeyValue> > &v_v_Instance, vector<int> &v_nLabel,
              vector<real> &classificationError){
    int batchSize = 10000;
    //create a miss labeling matrix for measuring the sub-classifier errors.
    model.missLabellingMatrix = vector<vector<int> >(model.nrClass, vector<int>(model.nrClass, 0));
    bool bEvaluateSubClass = true; //choose whether to evaluate sub-classifiers
    if(model.nrClass == 2)  //absolutely not necessary to evaluate sub-classifers
        bEvaluateSubClass = false;

    MultiPredictor predictor(model, model.param);

    clock_t start, finish;
    start = clock();
    int begin = 0;
    vector<int> predictLabels;
    while (begin < v_v_Instance.size()) {
        vector<real> decValue;
        //get a subset of instances
        int end = min(begin + batchSize, (int) v_v_Instance.size());
        vector<vector<KeyValue> > samples(v_v_Instance.begin() + begin,
                                          v_v_Instance.begin() + end);
        vector<int> vLabel(v_nLabel.begin() + begin, v_nLabel.begin() + end);
        if(bEvaluateSubClass == false)
            vLabel.clear();
        //predict labels for the subset of instances
       /* // procces for vote
	vector<int> predictLabelPart = predictor.predict(samples, vLabel);

        predictLabels.insert(predictLabels.end(), predictLabelPart.begin(), predictLabelPart.end());*/
        predictor.predictDecValue(decValue, samples);
        cout<<"decValue.size *****"<<decValue.size()<<endl;
		combDecValue.insert(combDecValue.end(), decValue.begin(), decValue.end());

        begin += batchSize;
    }
    finish = clock();
   /* process for vote 
    //combine bianry predictLabels
    combPredictLabels.push_back(predictLabels);*/
        
    if(bEvaluateSubClass == true){
        ClassifierEvaluater::evaluateSubClassifier(model.missLabellingMatrix, classificationError);
    }
    return (float) (finish - start) / CLOCKS_PER_SEC;
}


void evaluateOVAVote(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<int> > &combPredictLabels, vector<int> &originalPositiveLabel, float testingTime){    //read test set
      //vote for class
    int manyClassIns=0;//#instance that belong to more than one classes
    int NoClassIns=0;//#instance that does't belong to any class
    int correctIns=0;
	int nrClass=originalPositiveLabel.size();
	clock_t start,end;
	start=clock();
    for( int i=0;i<testInstance.size() ;i++){
        vector<int> vote(nrClass,0);
        int flag=0;
        int maxVote=0;
        for( int j=0;j<nrClass ;j++){
            if(combPredictLabels[j][i]==0)//if predictLabel=0 then instance belongs to the label 0 in jth bianrySVM
            {
		vote[j]++;
                flag++;
                maxVote=j;
			}
        }
			
		if(i<10)
    		cout<<"flaglabel********"<<flag<<endl;
        if(flag==1){
            if(originalPositiveLabel[maxVote]==testLabel[i])
                correctIns++;
			cout<<"flag==1"<<endl;	
				}
        else if(flag>1){
            manyClassIns++;
			cout<<"many"<<endl;}
        else{
            NoClassIns++;
			cout<<"noclass"<<endl;
			}
		if(i<10)
    		cout<<"manyclasslabel********"<<manyClassIns<<endl;
    }
	end=clock();
	testingTime+=(float)(end-start)/CLOCKS_PER_SEC;
    printf("classifier accuracy = %.2f%%(%d/%d)\n", correctIns / (float) testInstance.size() * 100,
           correctIns, (int) testInstance.size() );
    printf("number of unclaasifiable instances in OVA is %.2f%%(%d/%d)\n", manyClassIns / (float) testInstance.size() * 100, manyClassIns, testInstance.size());
    printf("number of NoClass instances in OVA is %.2f%%(%d/%d)\n", NoClassIns / (float) testInstance.size() * 100, NoClassIns, testInstance.size() );
    printf("prediction time elapsed: %.2fs\n",testingTime);

    
}

void evaluateOVADecValue(vector<vector<KeyValue> > &testInstance, vector<int> &testLabel, vector<vector<real> > &combDecValue, vector<int > originalPositiveLabel, float testingTime, ofstream &ofs){    //read test set
      //vote for class
    int manyClassIns=0;//#instance that belong to more than one classes
    int noClassIns=0;//#instance that does't belong to any class
    int correctIns=0;
	int nrClass=originalPositiveLabel.size();

    clock_t start,end;
    start=clock();
    for( int i=0;i<testInstance.size() ;i++){
        int flag=0;
        int max=0;
        for( int j=0;j<nrClass ;j++){
            if(combDecValue[j][i]>0&&combDecValue[j][i]>=combDecValue[max][i])//if predictLabel=0 then instance belongs to the label 0 in jth bianrySVM
            {
                flag++;
                max=j;
            }
        }
            
      
        if(flag>0){
            int manyClassflag=0;
            for(int j=0;j<nrClass ;j++){
                if(j!=max){
                    if(combDecValue[j][i]==combDecValue[max][i]){
                        manyClassIns++;
                        manyClassflag++;
                        break;
                    }
                }
                
            }
            if(manyClassflag==0){
                if (originalPositiveLabel[max]==testLabel[i])
                    correctIns++;
		if(i<10)
		    cout<<testLabel[i]<<endl;
            }
        }
        else
            noClassIns++;
    }
    end=clock();
    testingTime+=(float)(end-start)/CLOCKS_PER_SEC;
    printf("test  accuracy = %.2f%%(%d/%d)\n", correctIns / (float) testInstance.size() * 100,
           correctIns, (int) testInstance.size() );
    ofs<<"test accuracy: "<<correctIns / (float) testInstance.size()*100<<" ("<<correctIns<<"/"<<testInstance.size()<<")"<<"\n";
    
	printf("number of unclaasifiable instances in OVA is %.2f%%(%d/%d)\n", manyClassIns / (float) testInstance.size() * 100, manyClassIns, testInstance.size());
    ofs<<"test: # of manyClass instance: "<<manyClassIns / (float) testInstance.size()*100<<" ("<<manyClassIns<<"/"<<testInstance.size()<<")"<<"\n";
    
	printf("number of NoClass instances in OVA is %.2f%%(%d/%d)\n", noClassIns / (float) testInstance.size() * 100, noClassIns, testInstance.size() );
    ofs<<"test: #noClass instance:  "<<noClassIns / (float) testInstance.size()*100<<" ("<<noClassIns<<"/"<<testInstance.size()<<")"<<"\n";
    printf("prediction time elapsed: %.2fs\n",testingTime);
	ofs<<"testing time: "<<testingTime<<"\n";

}

void trainOVASVM(SVMParam &param, string strTrainingFileName, int numFeature,  bool evaluteTrainingError, string strTestingFileName, ofstream &ofs) {
    //nrclass must >2

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
    //build problem of all classes
    SvmProblem problem(v_v_Instance, numFeature, v_nLabel);
    int testNumInstance = 0;     //not used
    uint testNumofValue = 0;
    vector<vector<KeyValue> > testInstance;
    vector<int> testLabel;
    BaseLibSVMReader::GetDataInfo(strTestingFileName, numFeature, testNumInstance, testNumofValue);
    drHelper.ReadLibSVMAsSparse(testInstance, testLabel, strTestingFileName, numFeature);
    
	int nrClass=problem.getNumOfClasses();
    //vector<SvmModel> combModel(nrClass);
    vector<vector<int> > combPredictLabels;//combine k binary predictLaebl
    vector<vector<real> > combDecValue(nrClass);
    vector<vector<int> > combTrainPredictLabels;//combine k binary predictLaebl
    vector<int> originalPositiveLabel(nrClass);
    float testingTime=0;
    float allTrainingTime=0;
    float avgTrainingTime=0;
    for(int i=0;i<nrClass;i++)
        originalPositiveLabel[i]=v_nLabel[problem.perm[problem.start[i]]];
    
/*    vector<SVMParam> v_param(nrClass,param); 
	v_param[0].C=4;
	v_param[0].gamma=0.5;
	v_param[1].C=1;
	v_param[1].gamma=0.000244;
	v_param[2].C=2;
	v_param[2].gamma=0.001952;
	v_param[3].C=1;
	v_param[3].gamma=0.25;
	v_param[4].C=8;
	v_param[4].gamma=0.000488;
	v_param[5].C=4;
	v_param[5].gamma=0.00000;
*/
    //train and predict bianry svm
    for(int i=0;i<nrClass;i++){
        v_v_Instance=problem.v_vSamples;
        v_nLabel=problem.v_nLabels;
	SvmModel model;
        //reassign the 0 and 1 label to instances.
        for(int m=0;m<problem.count[i];m++)
            v_nLabel[problem.perm[problem.start[i] + m]]=0;//0 denotes the positive class
        for(int n=0;n<nrClass;n++){
            if(n!=i){
                for(int l=0;l<problem.count[n];l++)
                    v_nLabel[problem.perm[problem.start[n] + l]]=1;
            }
        }
	ofs<<"#i class instance "<<i<<": "<<problem.count[i]<<"\n";
        //for class i=0, in training phase, class i will be +1,
        //for other classes (i!=0), in training phase, class i will be -1
	//therfore, for class i!=0, swap the start[i] with the first instance in traing dataset, to make the class i to be the positive class in training phase
	if(i>0){
        vector<KeyValue> tempIns;
        int tempLabel;
	//swap label
	tempLabel=v_nLabel[problem.perm[problem.start[i]]];
        v_nLabel[problem.perm[problem.start[i]]]=v_nLabel[0];
        v_nLabel[0]=tempLabel;
	//swap instance 
        tempIns=v_v_Instance[problem.perm[problem.start[i]]];
        v_v_Instance[problem.perm[problem.start[i]]]=v_v_Instance[0];
        v_v_Instance[0]=tempIns;
      
    }
	for(int j=0;j<10;j++){
	cout<<"******************relabel 10 label"<<endl;
	cout<<"label "<<j<<":"<<v_nLabel[j];
}
        //use instance with label 0 and 1 to build the problem
        SvmProblem binaryProblem(v_v_Instance, numFeature, v_nLabel);
        //problem.label=model.label  label[0]=the label of the first instance.

        model.fit(binaryProblem, param);//resize nrclass!!!!!solve->getsubproblem!!!!
        PRINT_TIME("training", trainingTimer)
        PRINT_TIME("working set selection",selectTimer)
        PRINT_TIME("pre-computation kernel",preComputeTimer)
        PRINT_TIME("iteration",iterationTimer)
        PRINT_TIME("g value updating",updateGTimer)
        model.saveLibModel(strTrainingFileName,problem);//save model in the same format as LIBSVM's
		allTrainingTime+=trainingTimer.getTotalTime();
		avgTrainingTime+=trainingTimer.getAverageTime();
//    PRINT_TIME("2 instances selection",selectTimer)
//    PRINT_TIME("kernel calculation",calculateKernelTimer)
//    PRINT_TIME("alpha updating",updateAlphaTimer)
//    PRINT_TIME("init cache",initTimer)

        //evaluate training error
        if (evaluteTrainingError == true) {
            printf("Computing training accuracy...\n");
            evaluate(model, testInstance, testLabel, ClassifierEvaluater::trainingError, ofs);
        }

        cout << "start evaluation..." << endl;
        testingTime+= evaluateOVABinaryClassifier(combDecValue[i], combPredictLabels, model, problem.v_vSamples, problem.v_nLabels, ClassifierEvaluater::testingError);
        //testingTime+= evaluateOVABinaryClassifier(combDecValue[i], combPredictLabels, model, testInstance, testLabel, ClassifierEvaluater::testingError);
        //evaluateOVABinaryClassifier(combTrainPredictLabels, model, v_v_Instance, v_nLabel, ClassifierEvaluater::testingError);//evaluate binary svm for vote strategy
   
    }
    ofs<<"total training time "<<allTrainingTime<<" avg time"<<avgTrainingTime<<"\n";
	ofs<<"all evaluation"<<endl;
    evaluateOVADecValue(problem.v_vSamples, problem.v_nLabels, combDecValue, originalPositiveLabel, testingTime, ofs);    //read test set
    //evaluateOVADecValue(testInstance, testLabel, combDecValue, originalPositiveLabel, testingTime, ofs);    //read test set

	//evaluateOVAVote(testInstance, testLabel, combPredictLabels, originalPositiveLabel, testingTime);
	//evaluateOVA(v_v_Instance, v_nLabel, combTrainPredictLabels, originalPositiveLabel, testingTime);
}
