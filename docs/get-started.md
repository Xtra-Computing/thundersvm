Getting Started with ThunderSVM
===============================
Here we provide a quick start tutorial for users to install and test ThunderSVM.

## Prerequisites
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* cmake

## Installation
* Clone ThunderSVM repository
```bash
git clone git@github.com:zeyiwen/mascot.git
```
* Download testing datasets
```bash
cd mascot
dataset/get_datasets.sh
```
* Build binary for testing
```
mkdir build
cd build
cmake ..
make -j runtest
```
Make sure all the test cases pass.
