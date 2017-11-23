Interface for Matlab.
Example:
>> n = ["-c", "10", "-g", "0.125", "test_dataset.txt", "test_dataset.model"]
>> m = cellstr(n)
>> svm_train_matlab(m)
>> n = ["test_dataset.txt", "test_dataset.model", "test_datset.out"]
>> m = cellstr(n)
>> svm_predict_matlab(m)
