/*
 * svmMain.cu
 *
 *  Created on: May 21, 2012
 *      Author: Zeyi Wen
 */

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "cvFunction.h"
#include "SVMCmdLineParser.h"
#include "../SharedUtility/initCuda.h"
#include "svmModel.h"
#include "trainClassifier.h"
#include "classifierEvaluater.h"
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	char fileName[1024];
	char savedFileName[1024];
	SVMCmdLineParser parser;
	parser.ParseLine(argc, argv, fileName, savedFileName);

    CUcontext context;
	if(!InitCUDA(context, 'G'))
	
		return 0;
	printf("CUDA initialized.\n");

    if(parser.task_type == 0){
		//perform svm training
		cout << "performing training" << endl;
		SvmModel model;
		trainSVM(parser.param, fileName, parser.numFeature, model, parser.compute_training_error);
    }else if(parser.task_type == 1){
		//perform cross validation*/
		cout << "performing cross-validation" << endl;
		crossValidation(parser.param, fileName);
	}else if(parser.task_type == 2){
 		//perform svm evaluation 
		cout << "performing evaluation" << endl;
		SvmModel model;
        cout << "start training..." << endl;
		string name=fileName;
		name="./result/"+name;
		ofstream ofs(name.c_str(),ios::app);

		if(!ofs.is_open()){
			cout<<"open ./result/ error "<<name<<endl;
			return 0;
		}
		ofs<<"OVA"<<fileName<<"\n";
		ofs<<"g "<<parser.param.gamma<<"C"<<parser.param.C<<"\n";
		trainSVM(parser.param, fileName, parser.numFeature, model, parser.compute_training_error);
        cout << "start evaluation..." << endl;
        evaluateSVMClassifier(model, parser.testSetName, parser.numFeature);
	ofs<<"\n";
	ofs.close();
    }else if(parser.task_type == 4){
    	//perform selecting best C
    	cout << "perform C selection" << endl;
    	vector<real> vC;
    	for(int i = 0; i < 3; i++){
			SvmModel model;
			model.vC = vC;
			cout << "start training..." << endl;
			trainSVM(parser.param, fileName, parser.numFeature, model, parser.compute_training_error);
			cout << "start evaluation..." << endl;
			//evaluateSVMClassifier(model, strcat(fileName, ".t"), parser.numFeature);
			evaluateSVMClassifier(model, fileName, parser.numFeature);
			vC = ClassifierEvaluater::updateC(model.vC);
    	}
    }
	else if(parser.task_type == 3){
    	cout << "performing grid search" << endl;
    	Grid paramGrid;
    	paramGrid.vfC.push_back(parser.param.C);
    	paramGrid.vfGamma.push_back(parser.param.gamma);
    	gridSearch(paramGrid, fileName);
    }else if(parser.task_type == 5){
 		//perform svm evaluation 
		
		cout << "performing evaluation" << endl;
        cout << "start training..." << endl;
		string name=fileName;
		name="./result/"+name;
		ofstream ofs(name.c_str(),ios::app);

		if(!ofs.is_open()){
			cout<<"open ./result/ error "<<name<<endl;
			return 0;
		}
		ofs<<"OVA"<<fileName<<"\n";
		//double gamma=parser.param.gamma;
		//double C=parser.param.C;
    	//for(int grid=1; grid<3;grid*=2){
//			parser.param.gamma=gamma*grid*2;
//			for(int grid2=1;grid2<17;grid2*=2){
//			parser.param.C=C*grid2*2;
			ofs<<"g "<<parser.param.gamma<<"C"<<parser.param.C<<"\n";
			trainOVASVM(parser.param, fileName, parser.numFeature, parser.compute_training_error, parser.testSetName, ofs);
			ofs<<"\n";
		//}
//		}
		ofs.close();
    }else{
    	cout << "unknown task type" << endl;
    }

    return 0;
}
