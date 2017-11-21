Getting Started with ThunderSVM
===============================
Here we provide a quick start tutorial for users to install and test ThunderSVM.

## Prerequisites
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* cmake

## Installation
* Clone ThunderSVM repository
```bash
git clone --recursive git@github.com:zeyiwen/mascot.git
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

## Set running mode
We provide two ways to run ThunderSVM. By modifying CMakeList.txt in the thundersvm directory, you can choose whether or not to use gpu to speed up ThunderSVM. To run ThunderSVM with gpu, you should turn USE_CUDA on in CMakeList.txt.
```bash
set(USE_CUDA ON CACHE BOOL "Compile with CUDA")
```
To run ThunderSVM without gpu, you should turn USE_CUDA off.
```bash
set(USE_CUDA OFF CACHE BOOL "Compile with CUDA")
```

## Training SVMs
We show some concrete examples of using ThunderSVM. ThunderSVM uses the same command line options as LibSVM, so existing users of LibSVM can use ThunderSVM quickly. For new users of SVMs, the [user guide](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) provided in the LibSVM website also helps. 

### Training SVMs for Classification
In the following, we provide an example of using ThunderSVM for the MNIST dataset.

* Download the MNIST data set
The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2).

* Decompress the data set
```bash
bunzip2 mnist.scale.bz2
```

* Install ThunderSVM
Instructions available in [How To](how-to.md) page.

* Run ThunderSVM
```bash
./thundersvm -s 0 -t 2 -g 0.125 -c 10 mnist.scale svm.model
```
The meaning of each option can be found in the [parameters](parameters.md) page. Then you will see ThunderSVM automatically choose multi-class SVMs as the training algorithm. The training takes a while to complete. Once completed, you can see the training error is xx%.

### Training SVMs for Regression
The usage of other SVM algorithms (such as SVM regression) are similar to the above example. The key different is the selection of the options. Let's take Abalone data set as an example.

* Download the Abalone data set
The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale). You can use the following command to download.
```bash
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale
```

* Install ThunderSVM
Instructions available in [How To](how-to.md) page.

* Run ThunderSVM
```bash
./thundersvm -s 3 -t 2 -g 3.8 -c 1000 abalone_scale svm.model
```
The meaning of each option can be found in the [parameters](parameters.md) page. 

### Python Interface
ThunderSVM provides python interface. Go to the python subdirectory and type python. Put your dataset file in dataset subdirectory. Here is an example to call ThunderSVM functions.
```bash
>>>from svm import *
>>>y,x = svm_read_problem('mnist.scale')
>>>svm_train(y, x, 'mnist.scale.model', '-s 0 -t 2 -g 0.125 -c 10 -e 0.001')
>>>y,x = svm_read_problem('mnist.scale.t')
>>>svm_predict(y, x, 'mnist.scale.model', 'mnist.scale.out')
```
