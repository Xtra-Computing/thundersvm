# R interface for thundersvm.
Before you use the R interface, you must build ThunderSVM.

## Instructions for building ThunderSVM
* Please refer to [Installation](http://thundersvm.readthedocs.io/en/latest/how-to.html) for building ThunderSVM.

* Then, under ```./build/lib/``` of the ThunderSVM root directory, you should be able to see a library of ThunderSVM (e.g., ```libthundersvm.so``` on Linux machines).

* After you have successfully done the above two steps, it is ready to start using Python interfaces.
## Methods
By default, the directory for storing the training data and results is the working directory.

*svm_train_R(svm_type = 0, kernel = 2,degree = 3,gamma = 'auto',  coef0 = 0.0, nu = 0.5, cost = 1.0, epsilon = 0.1, tol = 0.001, probability = FALSE, class_weight = 'None', verbose = FALSE, max_iter = -1, dataset = 'None', model_file = 'None')*

*svm_predict_R(test_dataset = 'None', model_file = 'None', out_file = 'None')*

## Examples
```R
source("svm.R")
svm_train_R(dataset = "../dataset/test_dataset.txt", model_file = "../dataset/test_dataset.txt.model", cost = 100, gamma = 0.5)
svm_predict_R(test_dataset = "../dataset/test_dataset.txt", model_file = "../dataset/test_dataset.txt.model", out_file="test_dataset.txt.out")
```

## Parameters
*svm_type*: int, optional(default=0)
    set type of SVM (default 0)\
                    "	0 -- C-SVC		(multi-class classification)"\
                    "	1 -- nu-SVC		(multi-class classification)"\
                    "	2 -- one-class SVM"\
                    "	3 -- epsilon-SVR	(regression)"\
                    "	4 -- nu-SVR		(regression)"

*kernel*: int, optional(default=2)\
    set type of kernel function\
                    	0 -- linear: u'*v\
                    	1 -- polynomial: (gamma*u'*v + coef0)^degree\
                    	2 -- radial basis function: exp(-gamma*|u-v|^2)\
                    	3 -- sigmoid: tanh(gamma*u'*v + coef0)\
                    	4 -- precomputed kernel (kernel values in training_set_file)

*degree*: int, optional(default=3)\
    set degree in kernel function

*gamma*: float, optional(default='auto')\
    set gamma in kernel function (auto:1/num_features)

*coef0*: float, optional(default=0.0)\
    set coef0 in kernel function

*cost*: float, optional(default=1.0)\
    set the parameter C of C-SVC, epsilon-SVR, and nu-SVR

*nu*: float, optional(default=0.5)\
    set the parameter nu of nu-SVC, one-class SVM, and nu-SVR

*epsilon*: float, optional(default=0.1)\
    set the epsilon in loss function of epsilon-SVR

*tol*: float, optional(default=0.001)\
    set tolerance of termination criterion (default 0.001)

*probability*: boolean, optional(default=False)\
    whether to train a SVC or SVR model for probability estimates, True or False

*class_weight*:  {dict, ‘balanced’}, optional(default=None)\
    set the parameter C of class i to weight*C, for C-SVC

*verbose*: bool(default=False)\
    Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.

*max_iter*: int, optional (default=-1)\
    Hard limit on iterations within solver, or -1 for no limit.

*dataset*: string\
    dataset path to train.

*model_file*: string, optional\
    file path to save model.

*test_dataset*: string\
    dataset path to test

*out_file*: string, optional\
    file path to save predict outcomes.
