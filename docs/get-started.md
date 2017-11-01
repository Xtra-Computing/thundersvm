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

## Example of using ThunderSVM
ThunderSVM uses the same command line options as LibSVM, so existing users of LibSVM can use ThunderSVM quickly. For new users of SVMs, the [user guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) provided in the LibSVM website also helps. In the following, we provide an example of using ThunderSVM for the MNIST dataset.

* Download the MNIST data set
The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2).

* Decompress the data set
```bash
tar xvjf mnist.bz2
```

* Install ThunderSVM
Instructions available in [How To](how-to.md#install-thundersvm) page.