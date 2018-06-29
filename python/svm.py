#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
from os import path
import sys
from sys import platform

dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
    shared_library_name = "libthundersvm.so"
elif platform == "win32":
    shared_library_name = "thundersvm.dll"
elif platform == "darwin":
    shared_library_name = "libthundersvm.dylib"
else :
    print ("OS not supported!")
    exit()

if path.exists(path.abspath(path.join(dirname, shared_library_name))):
    lib_path = path.abspath(path.join(dirname, shared_library_name))
else:
    if platform == "linux" or platform == "linux2":
        lib_path = path.join(dirname, '../build/lib/', shared_library_name)
    elif platform == "win32":
        lib_path = path.join(dirname, '../build/bin/Debug/', shared_library_name)
    elif platform == "darwin":
        lib_path = path.join(dirname, '../build/lib/', shared_library_name)

if path.exists(lib_path):
    thundersvm = CDLL(lib_path)
else :
    print ("Please build the library first!")
    exit()
    
dataset_path = dirname
'''
class dataset(object):
    def __init__(self):
        self.obj = lib.DataSet_new()

    def load_from_python(self, arg1, arg2, arg3):
        lib.DataSet_load_from_python(self.obj, arg1, arg2, arg3)


def svm_train(param):
	param_list = param.split()
	param_list.insert(0, 'thundersvm-train')
	param_array = (c_char_p * len(param_list))()
	param_array[:] = param_list
	thundersvm.thundersvm_train(len(param_list), param_array)

def svm_predict(param):
	param_list = param.split()
	param_list.insert(0, 'thundersvm-predict')
	param_array = (c_char_p * len(param_list))()
	param_array[:] = param_list
	thundersvm.thundersvm_predict(len(param_list), param_array)
'''
def svm_read_problem(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	file_path = data_file_name
	for line in open(file_path):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = features.encode('utf-8')[:-1]
		#for e in features.split():
		#	ind, val = e.split(":")
		#	xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)

def svm_train(arg1, arg2 = None, arg3 = None, arg4 = None):
	if arg2:
		arg1_array = (c_float * len(arg1))()
		arg1_array[:] = arg1
		#arg2_string_list = [str(d).encode('utf-8')[1:-1] for d in arg2]
		arg2_string_list = arg2
		arg2_array = (c_char_p * len(arg2_string_list))()
		arg2_array[:] = arg2_string_list
		#print(arg1_array[0])
		#print(arg2_array[0])
		arg4_list = arg4.encode('utf-8').split()
		arg4_array = (c_char_p * len(arg4_list))()
		arg4_array[:] = arg4_list
		#dataset_python = dataset();
		#dataset_python.load_from_python(arg1, arg2, arg3)
		#print(dataset_python)
		thundersvm.load_from_python_interface(arg1_array, arg2_array, len(arg1_array))
		thundersvm.thundersvm_train_after_parse(arg4_array, len(arg4_array), arg3.encode('utf-8'))
	else:
		param_list = arg1.encode('utf-8').split()
		param_list.insert(0, 'thundersvm-train')
		param_array = (c_char_p * len(param_list))()
		param_array[:] = param_list
		#print(param_array[0])
		thundersvm.thundersvm_train(len(param_list), param_array)

def svm_predict(arg1, arg2 = None, arg3 = None, arg4 = None, arg5 = None):
	if arg2:
		arg1_array = (c_float * len(arg1))()
		arg1_array[:] = arg1
		arg2_array = (c_char_p * len(arg2))()
		arg2_array[:] = arg2
		thundersvm.load_from_python_interface(arg1_array, arg2_array, len(arg1_array))
		if arg5:
			arg5_list = arg5.encode('utf-8').split()
			arg5_array = (c_char_p * len(arg5_list))()
			arg5_array[:] = arg5_list
			thundersvm.thundersvm_predict_after_parse(arg3.encode('utf-8'), arg4.encode('utf-8'), arg5_array, len(arg5_array))
		else :
			arg5_array = None
			thundersvm.thundersvm_predict_after_parse(arg3.encode('utf-8'), arg4.encode('utf-8'), arg5_array, 0)
	else:
		param_list = arg1.split()
		param_list.insert(0, 'thundersvm-predict')
		param_array = (c_char_p * len(param_list))()
		param_array[:] = param_list
		thundersvm.thundersvm_predict(len(param_list), param_array)
#libsvm.thundersvm_train(15, "./thundersvm-train -s 1 -t 2 -g 0.5 -c 100 -n 0.1 -e 0.001 dataset/test_dataset.txt dataset/test_dataset.txt.model");
