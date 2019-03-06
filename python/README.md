We provide both simple Python interface and scikit-learn wrapper interface. Before you use the Python interface, you must build ThunderSVM.

## Instructions for building ThunderSVM
* Please refer to [Installation](http://thundersvm.readthedocs.io/en/latest/how-to.html) for building ThunderSVM.

* Then, if you want to install the Python package, go to the project root directory and run:
```
cd python && python setup.py install
```

* However, you don't need to install the Python package in order to use it from Python. Thus, under ```./build/lib/``` of the ThunderSVM root directory, you should be able to see a library of ThunderSVM (e.g., ```libthundersvm.so``` on Linux machines).

* After you have successfully done the above two steps, it is ready to start using Python interfaces.

## Simple Python interface
### Methods
By default, the directory for storing the training data and results is the working directory; the ThunderSVM library (e.g., ```libthundersvm.so```) is stored in ```../build/lib``` of the current working directory.

*svm_read_problem('file_name')*:\
	read data from *file_name*.\
*return: (labels, instances)*

*svm_train(labels, instances, 'model_file_name', parameters)*:\
	train the SVM model and save the result to *model_file_name*.

*svm_predict(labels, instances, 'model_file_name', 'output_file_name', parameters)*:\
	use the SVM model saved in *model_file_name* to predict the labels of the given instances and store the results to *output_file_name*.

### Example
* Step 1: go to the Python interface.
```bash
# in thundersvm root directory
cd python
```
* Step 2: create a file called ```test.py``` which has the following content.
```python
from svm import *
y,x = svm_read_problem('../dataset/test_dataset.txt')
svm_train(y,x,'test_dataset.txt.model','-c 100 -g 0.5')
y,x=svm_read_problem('../dataset/test_dataset.txt')
svm_predict(y,x,'test_dataset.txt.model','test_dataset.predict')
```
* Step 3: run the python script.
```bash
python test.py
```
## Scikit-learn wrapper interface
### Prerequisites
* numpy
* scipy
* sklearn
### Usage
The usage of thundersvm scikit interface is similar to sklearn.svm.


##### SVM classification
*class SVC(kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, C = 1.0, tol = 0.001, probability = False, class_weight = None, shrinking = False, cache_size = None, verbose = False, max_iter = -1, n_jobs = -1, max_mem_size = -1, random_state = None, decision_function_shape = 'ovo')*

*class NuSVC(kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, nu = 0.5, tol = 0.001, probability = False, shrinking = False, cache_size = None, verbose = False, max_iter = -1, n_jobs = -1, max_mem_size = -1, random_state = None, decision_function_shape = 'ovo')*

##### One-class SVMs

*class OneClassSVM(kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, nu = 0.5, tol = 0.001, shrinking = False, cache_size = None, verbose = False, max_iter = -1, n_jobs = -1, max_mem_size = -1, random_state = None)*

##### SVM regression
*class SVR(kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, C = 1.0, epsilon = 0.1, tol = 0.001, probability = False, shrinking = False, cache_size = None, verbose = False, max_iter = -1, n_jobs = -1, max_mem_size = -1)*

*class NuSVR(kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0, nu = 0.5, C = 1.0, tol = 0.001, probability = False, shrinking = False,  cache_size = None, verbose = False, max_iter = -1, n_jobs = -1, max_mem_size = -1)*


### Parameters
*kernel*: string, optional(default='rbf')\
    set type of kernel function\
                    	'linear': u'\*v\
                    	'polynomial': (gamma\*u'\*v + coef0)^degree\
                    	'rbf': exp(-gamma\*|u-v|^2)\
                    	'sigmoid': tanh(gamma\*u'\*v + coef0)\
                    	'precomputed' -- precomputed kernel (kernel values in training_set_file)

*degree*: int, optional(default=3)\
    set degree in kernel function

*gamma*: float, optional(default='auto')\
    set gamma in kernel function (auto:1/num_features)

*coef0*: float, optional(default=0.0)\
    set coef0 in kernel function

*C*: float, optional(default=1.0)\
    set the parameter C of C-SVC, epsilon-SVR, and nu-SVR

*nu*: float, optional(default=0.5)\
    set the parameter nu of nu-SVC, one-class SVM, and nu-SVR

*epsilon*: float, optional(default=0.1)\
    set the epsilon in loss function of epsilon-SVR

*tol*: float, optional(default=0.001)\
    set tolerance of termination criterion (default 0.001)

*probability*: boolean, optional(default=False)\
    whether to train a SVC or SVR model for probability estimates, True or False

*class_weight*:  {dict}, optional(default=None)\
    set the parameter C of class i to weight*C, for C-SVC

*shrinking*: boolean, optional (default=False, not supported yet for True)\
    whether to use the shrinking heuristic.

*cache_size*: float, optional, not supported yet.\
    specify the size of the kernel cache (in MB).

*verbose*: bool(default=False)\
    enable verbose output. Note that this setting takes advantage of a per-process runtime setting; if enabled, ThunderSVM may not work properly in a multithreaded context.

*max_iter*: int, optional (default=-1)\
    hard limit on the number of iterations within the solver, or -1 for no limit.

*n_jobs*: int, optional (default=-1)\
    set the number of cpu cores to use, or -1 for maximum.

*max_mem_size*: int, optional (default=-1)\
	set the maximum memory size (MB) that thundersvm uses, or -1 for no limit.

*gpu_id*: int, optional (default=0)\
	set which gpu to use for training.

*decision_function_shape*: ‘ovo’, default=’ovo’, not supported yet for 'ovr'\
    only for classifier. Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).

*random_state*: int, RandomState instance or None, optional (default=None), not supported yet\
    The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

### Attributes
*support_vectors_*: array-like, shape = [n_SV, n_features]\
    support vectors.

*n_support_*: array-like, dtype=int32, shape = [n_class]\
    number of support vectors for each class.

*dual_coef_*: array, shape = [n_class-1, n_SV]\
    coefficients of the support vector in the decision function. For multiclass, coefficient for all 1-vs-1 classifiers. The layout of the coefficients in the multiclass case is somewhat non-trivial.

*coef_*: array, shape = [n_class * (n_class-1)/2, n_features]\
    Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.
    
*intercept_*: array, shape = [n_class * (n_class-1) / 2]\
    constants in decision function.

### Methods
By default, the ThunderSVM library (e.g., ```libthundersvm.so```) is stored in ```../build/lib``` of the current working directory.

*fit(X, y)*:\
Fit the SVM model according to the given training data.

*get_params([deep])*:\
Get parameters for this estimator.

*predict(X)*:\
Perform classification on samples in X.

*score(X, y)*:\
Returns the mean accuracy on the given test data and labels.

*set_params(\*\*params)*:\
Set the parameters of this estimator.

*decision_function(X)*:\
Return distance of the samples X to the separating hyperplane. Only for SVC, NuSVC and OneClassSVM.

*save_to_file(path)*:\
Save the model to the file path.

*load_from_file(path)*:\
Load the model from the file path.

### Example

* Step 1: go to the Python interface.
```bash
# in thundersvm root directory
cd python
```
* Step 2: create a file called ```sk_test.py``` which has the following content.
```python
from thundersvmScikit import *
from sklearn.datasets import *

x,y = load_svmlight_file("../dataset/test_dataset.txt")
clf = SVC(verbose=True, gamma=0.5, C=100)
clf.fit(x,y)

x2,y2=load_svmlight_file("../dataset/test_dataset.txt")
y_predict=clf.predict(x2)
score=clf.score(x2,y2)
clf.save_to_file('./model')

print "test score is ", score
```
* Step 3: run the python script.
```bash
python sk_test.py
```
