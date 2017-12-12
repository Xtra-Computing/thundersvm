[![Build Status](https://travis-ci.org/zeyiwen/thundersvm.svg?branch=master)](https://travis-ci.org/zeyiwen/thundersvm)
[![Documentation Status](https://readthedocs.org/projects/thundersvm/badge/?version=latest)](https://thundersvm.readthedocs.org)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

<div align="center">
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/logo.png" width="240" height="220" align=left/>
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/lang-logo.png" width="250" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/overall.png" width="250" height="200" align=left/>
</div>

## Overview
The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. ThunderSVM exploits GPUs and multi-core CPUs to achieve high efficiency. Key features of ThunderSVM are as follows.
* Support all functionalities of LibSVM such as one-class SVMs, SVC, SVR and probabilistic SVMs.
* Use same command line options as LibSVM.
* Support Python, R and Matlab interfaces.

**Why accelerate SVMs**: A [survey](https://www.kaggle.com/amberthomas/kaggle-2017-survey-results) conducted by Kaggle in 2017 shows that 26% of the data mining and machine learning practitioners are users of SVMs.

[Documentations](http://thundersvm.readthedocs.io) | [Installation](http://thundersvm.readthedocs.io/en/latest/how-to.html) | [API Reference (doxygen)](http://zeyiwen.github.io/thundersvm/)
## Contents
- [Getting Started](https://github.com/zeyiwen/thundersvm#getting-started)
- [Working without GPUs](http://thundersvm.readthedocs.io/en/latest/get-started.html#working-without-gpus)
## Getting Started
### Prerequisites
* Supported Operating Systems: Linux, Windows and MacOS
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

## How to cite ThunderSVM
If you use ThunderSVM in your paper, please cite our work ([preprint now available](http://www.comp.nus.edu.sg/~hebs/pub/thundersvm.pdf)).
```
@article{wenthundersvm17,
 author = {Wen, Zeyi and Shi, Jiashuai and He, Bingsheng and Li, Qinbin and Chen, Jian},
 title = {{ThunderSVM}: A Fast SVM library on GPUs and CPUs},
 journal = {To appear in arxiv},
 year = {2017}
}
```
## Related websites
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) | [SVM<sup>light</sup>](http://svmlight.joachims.org/) | [OHD-SVM](https://github.com/OrcusCZ/OHD-SVM) | [NVIDIA Machine Learning](http://www.nvidia.com/object/machine-learning.html)

## Acknowlegement 
* We acknowledge NVIDIA for their hardware donations.
* This project is hosted by NUS, collaborating with Prof. Jian Chen (South China University of Technology). Initial work of this project was done when Zeyi Wen worked at The University of Melbourne.
* This work is partially supported by a MoE AcRF Tier 1 grant (T1 251RES1610) in Singapore.
* We also thank the authors of LibSVM and OHD-SVM which inspire our algorithmic design.
