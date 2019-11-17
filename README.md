[![Build Status](https://travis-ci.org/Xtra-Computing/thundersvm.svg?branch=master)](https://travis-ci.org/zeyiwen/thundersvm)
[![Build status](https://ci.appveyor.com/api/projects/status/e9yoehx7orsrsh89/branch/master?svg=true)](https://ci.appveyor.com/project/shijiashuai/thundersvm/branch/master)
[![GitHub license](https://img.shields.io/badge/license-apache2-yellowgreen)](./LICENSE)
[![Documentation Status](https://readthedocs.org/projects/thundersvm/badge/?version=latest)](https://thundersvm.readthedocs.org)
[![GitHub issues](https://img.shields.io/github/issues/Xtra-Computing/thundersvm.svg)](https://github.com/zeyiwen/thundersvm/issues)
[![PyPI version](https://badge.fury.io/py/thundersvm.svg)](https://badge.fury.io/py/thundersvm)

<div align="center">
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/logo.png" width="240" height="220" align=left/>
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/lang-logo.png" width="250" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundersvm/raw/master/docs/_static/overall.png" width="250" height="200" align=left/>
</div>

## What's new
- We have recently released [ThunderGBM](https://github.com/Xtra-Computing/thundergbm), a fast GBDT and Random Forest library on GPUs.
- add scikit-learn interface, see [here](https://github.com/Xtra-Computing/thundersvm/tree/master/python)
- pre-built binaries and DLL for Windows x64 on CPUs are [avaliable](https://ci.appveyor.com/project/shijiashuai/thundersvm/branch/master/artifacts)
## Overview
The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. ThunderSVM exploits GPUs and multi-core CPUs to achieve high efficiency. Key features of ThunderSVM are as follows.
* Support all functionalities of LibSVM such as one-class SVMs, SVC, SVR and probabilistic SVMs.
* Use same command line options as LibSVM.
* Support [Python](python/), [R](R/) and [Matlab](Matlab/) interfaces.
* Supported Operating Systems: Linux, Windows and MacOS.

**Why accelerate SVMs**: A [survey](https://www.kaggle.com/amberthomas/kaggle-2017-survey-results) conducted by Kaggle in 2017 shows that 26% of the data mining and machine learning practitioners are users of SVMs.

[Documentations](docs/index.md) | [Installation](docs/how-to.md#install-thundersvm) | [API Reference (doxygen)](http://Xtra-Computing.github.io/thundersvm/)
## Contents
- [Getting Started](https://github.com/Xtra-Computing/thundersvm#getting-started)
- [Working without GPUs](docs/get-started.md#working-without-gpus)
## Getting Started

### Prerequisites
* cmake 2.8 or above 
* gcc 4.8 or above for Linux and MacOS
* Visual C++ for Windows

If you want to use GPUs, you also need to install CUDA.

* [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5 or above

### Quick Install
Download the Python wheel file (For Python3 or above).

* For Linux

    * `pip install thundersvm` for CUDA 9.0 - linux_x86_64 
    
    * [CPU - linux_x86_64](https://github.com/Xtra-Computing/thundersvm/blob/d38af58e0ceb7e5d948f3ef7d2c241ba50133ee6/python/dist/thundersvm-cpu-0.2.0-py3-none-linux_x86_64.whl)

* For Windows (64bit)
    
    * [CUDA 10.0 - win64](https://github.com/Xtra-Computing/thundersvm/blob/d38af58e0ceb7e5d948f3ef7d2c241ba50133ee6/python/dist/thundersvm-cu10-0.2.0-py3-none-win_amd64.whl)
    
    * [CPU - win64](https://github.com/Xtra-Computing/thundersvm/blob/d38af58e0ceb7e5d948f3ef7d2c241ba50133ee6/python/dist/thundersvm-cpu-0.2.0-py3-none-win_amd64.whl)

Install the Python wheel file.
```bash
pip install thundersvm-cu90-0.2.0-py3-none-linux_x86_64.whl
```
##### Example
```python
from thundersvm import SVC
clf = SVC()
clf.fit(x, y)
```
### Download
```bash
git clone https://github.com/Xtra-Computing/thundersvm.git
```
### Build on Linux (build [instructions](docs/get-started.md#installation-for-macos) for MacOS and Windows)
##### ThunderSVM on GPUs
```bash
cd thundersvm
mkdir build && cd build && cmake .. && make -j
```

If you run into issues that can be traced back to your version of gcc, use `cmake` with a version flag to force gcc 6. That would look like this:

```bash
cmake -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 ..
```


##### ThunderSVM on CPUs
```bash
# in thundersvm root directory
git submodule init eigen && git submodule update
mkdir build && cd build && cmake -DUSE_CUDA=OFF -DUSE_EIGEN=ON .. && make -j
```
If ```make -j``` doesn't work, please simply use ```make```. The number of CPU cores to use can be specified by the ```-o``` option (e.g., ```-o 10```), and refer to [Parameters](docs/parameters.md) for more information.

### Quick Start
```bash
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/test_dataset.txt
./bin/thundersvm-predict ../dataset/test_dataset.txt test_dataset.txt.model test_dataset.predict
```
You will see `Accuracy = 0.98` after successful running.

## How to cite ThunderSVM
If you use ThunderSVM in your paper, please cite our work ([full version](https://github.com/zeyiwen/thundersvm/blob/master/thundersvm-full.pdf)).
```
@article{wenthundersvm18,
 author = {Wen, Zeyi and Shi, Jiashuai and Li, Qinbin and He, Bingsheng and Chen, Jian},
 title = {{ThunderSVM}: A Fast {SVM} Library on {GPUs} and {CPUs}},
 journal = {Journal of Machine Learning Research},
 volume={19},
 pages={797--801},
 year = {2018}
}
```

## Other publications
* Zeyi Wen, Jiashuai Shi, Bingsheng He, Yawen Chen, and Jian Chen. Efficient Multi-Class Probabilistic SVMs on GPUs. IEEE Transactions on Knowledge and Data Engineering (TKDE), 2018.
* Zeyi Wen, Bingsheng He, Kotagiri Ramamohanarao, Shengliang Lu, and Jiashuai Shi. Efficient Gradient Boosted Decision Tree Training on GPUs. The 32nd IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 234-243, 2018.

## Related websites
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) | [SVM<sup>light</sup>](http://svmlight.joachims.org/) | [OHD-SVM](https://github.com/OrcusCZ/OHD-SVM) | [NVIDIA Machine Learning](http://www.nvidia.com/object/machine-learning.html) | [ThunderGBM](https://github.com/Xtra-Computing/thundergbm)

## Acknowlegement 
* We acknowledge NVIDIA for their hardware donations.
* This project is hosted by NUS, collaborating with Prof. Jian Chen (South China University of Technology). Initial work of this project was done when Zeyi Wen worked at The University of Melbourne.
* This work is partially supported by a MoE AcRF Tier 1 grant (T1 251RES1610) in Singapore.
* We also thank the authors of LibSVM and OHD-SVM which inspire our algorithmic design.

## Selected projects that use ThunderSVM
 * Scene Graphs for Interpretable Video Anomaly Classification (published in NeurIPS18)
 * 3D Semantic Segmentation for High-resolution Aerial Survey Derived Point Clouds using Deep Learning (published in SIGSPATIAL’18)
 * Accounting for part pose estimation uncertainties during trajectory generation for part pick-up using mobile manipulators. (published in International Conference on Robotics and Automation (ICRA), 2019).

