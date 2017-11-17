#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
from os import path
import sys


dirname = path.dirname(path.abspath(__file__))
libsvm = CDLL(path.join(dirname, '../build/lib/libthundersvm-lib.so'))

def svm_train(param):
	param_list = param.split()
	param_list.insert(0, 'thundersvm-train')
	param_array = (c_char_p * len(param_list))()
	param_array[:] = param_list
	libsvm.thundersvm_train(len(param_list), param_array)

def svm_predict(param):
	param_list = param.split()
	param_list.insert(0, 'thundersvm-predict')
	param_array = (c_char_p * len(param_list))()
	param_array[:] = param_list
	libsvm.thundersvm_predict(len(param_list), param_array)

#libsvm.thundersvm_train(15, "./thundersvm-train -s 1 -t 2 -g 0.5 -c 100 -n 0.1 -e 0.001 dataset/test_dataset.txt dataset/test_dataset.txt.model");
