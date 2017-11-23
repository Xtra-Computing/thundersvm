[![Build Status](https://travis-ci.org/zeyiwen/thundersvm.svg?branch=master)](https://travis-ci.org/zeyiwen/thundersvm)
[![Documentation Status](https://readthedocs.org/projects/thundersvm/badge/?version=latest)](https://thundersvm.readthedocs.org)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

<div align="center">
<img src="https://github.com/zeyiwen/thundersvm/raw/master/logo.png" width="240" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/lang-logo.png" width="250" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/overall.png" width="250" height="200" align=left/>
</div>

## Overview
The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. ThunderSVM exploits GPUs and multi-core CPUs to achieve high efficiency. Key features of ThunderSVM are as follows.
* Support all functionalities of LibSVM such as one-class SVMs, SVC, SVR and probabilistic SVMs.
* Use same command line options as LibSVM.
* Support Python, R and Matlab interfaces.

[Documentations](http://thundersvm.readthedocs.io) | [Installation](http://thundersvm.readthedocs.io/en/latest/how-to.html) | [API Reference (doxygen)](http://zeyiwen.github.io/thundersvm/)
## Contents
- [Getting Started](https://github.com/zeyiwen/thundersvm#getting-started)
- [Advanced](https://github.com/zeyiwen/thundersvm#advanced)
- [Working without GPUs](https://github.com/zeyiwen/thundersvm#working-withour-gpus)
## Getting Started
### Prerequisites
* [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5 or above | cmake 2.8 or above | gcc 4.8 or above
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
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/test_dataset.txt
./bin/thundersvm-predict ../dataset/test_dataset.txt test_dataset.txt.model test_dataset.predict
```
You will see `Accuracy = 0.98` after successful running.

## Advanced
## Working without GPUs
If you don't have GPUs, ThunderSVM can work with CPU only.
### Get Eigen Library
ThunderSVM uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix calculation. To use Eigen, just 
initialize the submodule. 
```bash
# in thundersvm root directory
git submodule init eigen && git submodule update
```
### Build without GPUs
```bash
# in thundersvm root directory
mkdir build && cd build && cmake -DUSE_CUDA=OFF -DUSE_EIGEN=ON .. && make -j
```
Now ThunderSVM will work solely on CPUs and does not rely on CUDA.

## Related websites
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) | [SVM<sup>light</sup>](http://svmlight.joachims.org/) | [OHD-SVM](https://github.com/OrcusCZ/OHD-SVM) | [NVIDIA Machine Learning](http://www.nvidia.com/object/machine-learning.html)

## TODO
- integrate with interfaces

## Acknowlegement 
* We acknowledge NVIDIA for their hardware donations.
* This project is hosted by NUS, collaborating with Prof. Jian Chen (South China University of Technology). Initial work of this project was done when Zeyi Wen worked at The University of Melbourne.
