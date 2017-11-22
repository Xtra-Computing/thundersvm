dyn.load("../build/lib/libthundersvm-lib.so")
svm_train_R <-
function(a1)
{
	a1<-c("thundersvm-train", a1)
	.C("thundersvm_train_R", as.integer(length(a1)), as.character(a1))
}

svm_predict_R <-
function(a1)
{
	a1<-c("thundersvm-predict", a1)
	.C("thundersvm_predict_R", as.integer(length(a1)), as.character(a1))
}