Interface for Matlab.
svm_train_matlab(parameters): The format of parameters is same as libsvm
	run svm train according to parameters.

svm_predict_matlab(parameters): The format of parameters is same as libsvm
	run svm predict according to parameters.
Example:
```matlab
n = ["-c", "10", "-g", "0.125", "test_dataset.txt", "test_dataset.model"]
m = cellstr(n)
svm_train_matlab(m)
n = ["test_dataset.txt", "test_dataset.model", "test_datset.out"]
m = cellstr(n)
svm_predict_matlab(m)
```
