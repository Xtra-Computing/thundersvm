[![Build Status](https://travis-ci.org/zeyiwen/thundersvm.svg?branch=master)](https://travis-ci.org/zeyiwen/thundersvm)

<div align="center"><img src="https://github.com/zeyiwen/thundersvm/raw/improve-doc/logo.png" width = "30%" height = "30%" align=left/>
</div>

# Overview
The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. Some key features of ThunderSVM are as follows.
* Support one-class, binary and multi-class SVM classification, SVM regression, and SVMs with probability outputs.
* Have Python, R and Matlab interfaces.

## Getting Started
### Prerequisites
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* cmake > 2.8
* gcc > 4.8
### Download
```bash
git clone git@github.com:zeyiwen/thundersvm.git
```
### Build
```bash
cd thundersvm
mkdir build && cd build && cmake .. && make -j
```
### Quick Start
```bash
bin\thundersvm-train -c 100 -g 0.5 ../dataset/test_dataset.txt
bin\thundersvm-predict ../dataset/test_dataset.txt test_dataset.model test_dataset.predict
```
### Run Tests
```bash
make runtest
```

## API
[API Reference](http://zeyiwen.github.io/thundersvm/)
## TODO
- integrate with interfaces

