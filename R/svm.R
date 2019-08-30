check_location <- function(){
  if(Sys.info()['sysname'] == 'Windows'){
    if(!file.exists("../build/bin/Debug/thundersvm.dll")){
      stop("Please build the library first!")
    }
    dyn.load("../build/bin/Debug/thundersvm.dll")
  } else if(Sys.info()['sysname'] == 'Linux'){
    if(!file.exists("../build/lib/libthundersvm.so")){
      stop("Please build the library first!")
    }
    dyn.load("../build/lib/libthundersvm.so")
  } else if(Sys.info()['sysname'] == 'Darwin'){
    if(!file.exists("../build/lib/libthundersvm.dylib")){
      stop("Please build the library first!")
    }
    dyn.load("../build/lib/libthundersvm.dylib")
  } else{
    stop("OS not supported!")
  }
}
check_location() # Run this when the file is sourced
	
svm_train_R <-
function(
svm_type = 0, kernel = 2,degree = 3,gamma = 'auto',
coef0 = 0.0, nu = 0.5, cost = 1.0, epsilon = 0.1,
tol = 0.001, probability = FALSE, class_weight = 'None', cv = '-1',
verbose = FALSE, max_iter = -1, n_cores = -1, dataset = 'None', model_file = 'None'
)
{
	check_location()
	if(!file.exists(dataset)){stop("The file containing the training dataset provided as an argument in 'dataset' does not exist")}
	res <- .C("train_R", as.character(dataset), as.integer(kernel), as.integer(svm_type),
	as.integer(degree), as.character(gamma), as.double(coef0), as.double(nu),
	as.double(cost), as.double(epsilon), as.double(tol), as.integer(probability),
	as.character(class_weight), as.integer(length(class_weight)), as.integer(cv),
	as.integer(verbose), as.integer(max_iter), as.integer(n_cores), as.character(model_file))
}

svm_predict_R <-
function(
test_dataset = 'None', model_file = 'None', out_file = 'None'
)
{
	check_location()
	if(!file.exists(test_dataset)){stop("The file containing the training dataset provided as an argument in 'test_dataset' does not exist")}
	if(!file.exists(model_file)){stop("The file containing the model provided as an argument in 'model_file' does not exist")}
	res <- .C("predict_R", as.character(test_dataset), as.character(model_file), as.character(out_file))
}
