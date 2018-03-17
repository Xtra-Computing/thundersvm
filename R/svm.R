if(!file.exists("../build/lib/libthundersvm.so")){
	print("Please build the library first!")
	quit()
}
dyn.load("../build/lib/libthundersvm.so")
svm_train_R <-
function(
svm_type = 0, kernel = 2,degree = 3,gamma = 'auto',
coef0 = 0.0, nu = 0.5, cost = 1.0, epsilon = 0.1,
tol = 0.001, probability = FALSE, class_weight = 'None',
verbose = FALSE, max_iter = -1, n_cores = -1, dataset = 'None', model_file = 'None'
)
{
	res <- .C("train_R", as.character(dataset), as.integer(kernel), as.integer(svm_type),
	as.integer(degree), as.character(gamma), as.double(coef0), as.double(nu),
	as.double(cost), as.double(epsilon), as.double(tol), as.integer(probability),
	as.character(class_weight), as.integer(length(class_weight)),
	as.integer(verbose), as.integer(max_iter), as.integer(n_cores), as.character(model_file))
}

svm_predict_R <-
function(
test_dataset = 'None', model_file = 'None', out_file = 'None'
)
{
	res <- .C("predict_R", as.character(test_dataset), as.character(model_file), as.character(out_file))
}