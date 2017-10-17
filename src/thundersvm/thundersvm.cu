//
// Created by jiashuai on 17-9-14.
//


#include <thundersvm/util/log.h>
#include <thundersvm/model/svc.h>
#include <thundersvm/model/svr.h>
#include <thundersvm/model/oneclass_svc.h>
#include "thundersvm/cmdparser.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {
	CMDParser parser;
	parser.parse_command_line(argc, argv);

    if(parser.param_cmd.task_type == svmTrain){
		//perform svm training
		printf("performing training\n");
	    DataSet train_dataset;
        train_dataset.load_from_file(parser.svmtrain_input_file_name);
        SvmModel *model;
        if (parser.param_cmd.svm_type == SvmParam::C_SVC)
        	model = new SVC();
        else if (parser.param_cmd.svm_type == SvmParam::EPSILON_SVR)
        	model = new SVR();
        else if (parser.param_cmd.svm_type == SvmParam::ONE_CLASS)
        	model = new OneClassSVC();
        else{
        	printf("the svm type is not supported yet!\n");
        	exit(-1);
        }

        if(parser.do_cross_validation == true){
        	model->cross_validation(train_dataset, parser.param_cmd, parser.nr_fold);
        	return 0;
        }
        else
        	model->train(train_dataset, parser.param_cmd);

        //perform svm testing
        DataSet test_dataset;
	    vector<real> predict_y;
        test_dataset.load_from_file(parser.svmtrain_input_file_name);

        predict_y = model->predict(test_dataset.instances(), 10000);
        if (parser.param_cmd.svm_type == SvmParam::C_SVC) {
			int n_correct = 0;
			for (int i = 0; i < predict_y.size(); ++i) {
				if (predict_y[i] == test_dataset.y()[i])
					n_correct++;
			}
			float accuracy = n_correct / (float) test_dataset.instances().size();
			LOG(INFO) << "Accuracy = " << accuracy << "(" << n_correct << "/"
					  << test_dataset.instances().size() << ")\n";
        } else if (parser.param_cmd.svm_type == SvmParam::EPSILON_SVR) {
			real mse = 0;
			for (int i = 0; i < predict_y.size(); ++i) {
				mse += (predict_y[i] - train_dataset.y()[i]) * (predict_y[i] - train_dataset.y()[i]);
			}
			mse /= predict_y.size();

			LOG(INFO) << "MSE = " << mse;
        } else if (parser.param_cmd.svm_type == SvmParam::ONE_CLASS) {
            vector<real> predict_y = model->predict(train_dataset.instances(), 100);
            int n_pos = 0;
            for (int i = 0; i < predict_y.size(); ++i) {
                if (predict_y[i] > 0)
                    n_pos++;
            }
            LOG(INFO) << "n_pos = " << n_pos;
        }
    }
    else{
    	printf("unknown task type\n");
    }
    return 0;
}

