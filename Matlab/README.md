Interface for Matlab.

### Instructions for building ThunderSVM
* Please refer to [Installation](http://thundersvm.readthedocs.io/en/latest/how-to.html) for building ThunderSVM.

* Then, under ```./build/lib/``` of the ThunderSVM root directory, you should be able to see a library of ThunderSVM (e.g., ```libthundersvm.so``` on Linux machines).

* After you have successfully done the above two steps, it is ready to start using this Matlab interface.

### Methods
*svm_train_matlab(parameters)*: The format of parameters is the same as libsvm
	train svm using the parameters.

*svm_predict_matlab(parameters)*: The format of parameters is the same as libsvm
	run svm predict using the parameters.

### Examples
```Matlab
n = ["-c", "10", "-g", "0.125", "test_dataset.txt", "test_dataset.model"]
m = cellstr(n)
svm_train_matlab(m)
n = ["test_dataset.txt", "test_dataset.model", "test_datset.out"]
m = cellstr(n)
svm_predict_matlab(m)
```
