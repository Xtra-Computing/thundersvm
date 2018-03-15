Getting Started with ThunderSVM
===============================
Here we provide a quick start tutorial for users to install ThunderSVM.

## Prerequisites
* ```cmake``` 2.8 or above
* ```gcc``` 4.8 or above for Linux and MacOS; ```Visual C++``` for Windows

If you want to use GPUs, you also need to install CUDA.

* [CUDA](https://developer.nvidia.com/cuda-downloads)

## Installation
* Clone ThunderSVM repository
```bash
git clone git@github.com:zeyiwen/thundersvm.git
```

* Download testing datasets

In Linux or MacOS, use the following commands.
```bash
cd thundersvm
dataset/get_datasets.sh
```
In Windows, you need to install tools such as [Cygwin](https://www.cygwin.com/) to run ```dataset/get_datasets.sh```. Alternatively, you can manually download the data sets listed in ```dataset/get_datasets.sh```.

* Build the binary for testing 

For Linux and MacOS, use the following commands.
```bash
mkdir build
cd build
cmake ..
make -j runtest
```
If ```make -j runtest``` doesn't work, please use ```make runtest``` instead.

For Windows, use the following example commands. You need to change the Visual Studio version if you are using a different version of Visual Studio. Visual Studio can be downloaded from [this link](https://www.visualstudio.com/vs/).
```bash
mkdir build
cd build
cmake .. -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE -G "Visual Studio 14 2015 Win64"
```
The above commands generate some Visual Studio project files, open the Visual Studio project in the build directory to start building ```runtest``` on Windows. Please note that CMake should be 3.4 or above for Windows.

Make sure all the test cases pass.

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

## Training SVMs
We show some concrete examples of using ThunderSVM. ThunderSVM uses the same command line options as LibSVM, so existing users of LibSVM can use ThunderSVM quickly. For new users of SVMs, the [parameters](parameters.md) page provides explanation for the usage of each option. 

### Training SVMs for Classification
In the following, we provide an example of using ThunderSVM for the MNIST dataset.

* Download the MNIST data set

The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2).

* Decompress the data set

For Unix based OSes, you can use
```bash
bunzip2 mnist.scale.bz2
```
For Windows machines, you can decompress the data set using tools such as [7-Zip](www.7-zip.org).

* Install ThunderSVM

Instructions available in [How To](how-to.md) page.

* Run ThunderSVM
```bash
./thundersvm-train -s 0 -t 2 -g 0.125 -c 10 mnist.scale svm.model
```
The meaning of each option can be found in the [parameters](parameters.md) page. The training takes a while to complete. Once completed, you can see the classifier accuracy is 94.32%.

### Training SVMs for Regression
The usage of other SVM algorithms (such as SVM regression) are similar to the above example. The key different is the selection of the options. Let's take Abalone data set as an example.

* Download the Abalone data set

The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale).

* Install ThunderSVM

Instructions available in [How To](how-to.md) page.

* Run ThunderSVM
```bash
./thundersvm-train -s 3 -t 2 -g 3.8 -c 1000 abalone_scale svm.model
```
The meaning of each option can be found in the [parameters](parameters.md) page. 

### Interfaces
ThunderSVM provides python, R and Matlab interface. You can find the instructions in the corresponding subdirectories.
