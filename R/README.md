Interface for R.
svm_train_R(parameters): The format of parameters is same as libsvm
	run svm train according to parameters.

svm_predict_R(parameters): The format of parameters is same as libsvm
	run svm predict according to parameters.

Example:
> source("svm.R")
> c<-c("-c", "10", "-g", "0.125", "test_dataset.txt", "test_dataset.model")
> svm_train_R(c)
> c<-c("test_dataset.txt","test_dataset.model","test_dataset.out")
> svm_predict_R(c)
