svm_read_problem('file_name'):
	read data from file_name in dataset directory.
return: (labels, instances)

svm_train(labels, instances, 'model_file_name', parameters):
	train (labels, instances) according to parameters and save the result to model_file_name in dataset directory.

svm_predict(labels, instances, 'model_file_name', 'output_file_name', parameters):
	use model_file_name to predict (labes, instances) according to parameters and save the result to output_file_name in dataset directory.

Example:
>>> from svm import *
>>> y,x = svm_read_problem('mnist.scale')
>>> svm_train(y,x,'mnist.scale.model','-s 0 -t 2 -g 0.125 -c 10 -e 0.001')
>>> y,x=svm_read_problem('mnist.scale.t')
>>> svm_predict(y,x,'mnist.scale.model','mnist.scale.out')

