We provide simple python interface and Scikit-Learn Wrapper interface.

1.simple python interface:
svm_read_problem('file_name'):
	read data from file_name in dataset directory.
return: (labels, instances)

svm_train(labels, instances, 'model_file_name', parameters):
	train (labels, instances) according to parameters and save the result to model_file_name in dataset directory.

svm_predict(labels, instances, 'model_file_name', 'output_file_name', parameters):
	use model_file_name to predict (labes, instances) according to parameters and save the result to output_file_name in dataset directory.

Example:
```python
from svm import *
y,x = svm_read_problem('mnist.scale')
svm_train(y,x,'mnist.scale.model','-s 0 -t 2 -g 0.125 -c 10 -e 0.001')
y,x=svm_read_problem('mnist.scale.t')
svm_predict(y,x,'mnist.scale.model','mnist.scale.out')
```

2.Scikit-Learn Wrapper interface:
The usage of thundersvm scikit interface is similar to scikit.svm

class SVC(kernel = 2, degree = 3,
          gamma = 'auto', coef0 = 0.0, cost = 1.0,
          tol = 0.001, probability = False, class_weight = None)

class NuSVC(kernel = 2, degree = 3, gamma = 'auto',
            coef0 = 0.0, nu = 0.5, tol = 0.001,
            probability = False)

class OneClassSVM(kernel = 2, degree = 3, gamma = 'auto',
                  coef0 = 0.0, nu = 0.5, tol = 0.001)

class SVR(kernel = 2, degree = 3, gamma = 'auto',
          coef0 = 0.0, cost = 1.0, epsilon = 0.1,
          tol = 0.001, probability = False)

class NuSVR(kernel = 2, degree = 3, gamma = 'auto',
            coef0 = 0.0, cost = 1.0, tol = 0.001, probability = False)


Parameters:
kernel: int, optional(default=2)
    set type of kernel function
                    	0 -- linear: u'*v
                    	1 -- polynomial: (gamma*u'*v + coef0)^degree
                    	2 -- radial basis function: exp(-gamma*|u-v|^2)
                    	3 -- sigmoid: tanh(gamma*u'*v + coef0)
                    	4 -- precomputed kernel (kernel values in training_set_file)

degree: int, optional(default=3)
    set degree in kernel function

gamma: float, optional(default='auto')
    set gamma in kernel function (auto:1/num_features)

coef0: float, optional(default=0.0)
    set coef0 in kernel function

cost: float, optional(default=1.0)
    set the parameter C of C-SVC, epsilon-SVR, and nu-SVR

nu: float, optional(default=0.5)
    set the parameter nu of nu-SVC, one-class SVM, and nu-SVR

epsilon: float, optional(default=0.1)
    set the epsilon in loss function of epsilon-SVR

tol: float, optional(default=0.001)
    set tolerance of termination criterion (default 0.001)

probability: boolean, optional(default=False)
    whether to train a SVC or SVR model for probability estimates, True or False

class_weight:  {dict, ‘balanced’}, optional(default=None)
    set the parameter C of class i to weight*C, for C-SVC



examples:
```python
from thundersvmScikit import *
from sklearn.datasets import *
x,y = load_svmlight_file("path to training dataset")
clf = SVC()
clf.fit(x,y)

x2,y2=load_svmlight_file("path to test dataset")
clf.predict(x2)
clf.score(x2,y2)

from sklearn.model_selection import *
scores = cross_val_score(clf,x,y,cv=5)
```

methods:
fit(X, y)	Fit the SVM model according to the given training data.
get_params([deep])	Get parameters for this estimator.
predict(X)	Perform classification on samples in X.
score(X, y)	Returns the mean accuracy on the given test data and labels.
set_params(**params)	Set the parameters of this estimator.



