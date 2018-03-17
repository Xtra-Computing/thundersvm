Getting Started with ThunderSVM
===============================
Here we provide a quick start tutorial for users to install ThunderSVM.

## Prerequisites
* ```cmake``` 2.8 or above
* ```gcc``` 4.8 or above for Linux and MacOS; ```Visual C++``` for Windows

If you want to use GPUs, you also need to install CUDA.

* [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5 or above

## Installation
#### Installation for  Linux
* Clone ThunderSVM repository
```bash
git clone git@github.com:zeyiwen/thundersvm.git
```
* Download testing datasets
```bash
cd thundersvm
dataset/get_datasets.sh
```

* Build the binary for testing 
```bash
mkdir build
cd build
cmake ..
make -j runtest
```
If ```make -j runtest``` doesn't work, please use ```make runtest``` instead. Make sure all the test cases pass.

#### Installation for MacOS
* Clone ThunderSVM repository
```bash
git clone git@github.com:zeyiwen/thundersvm.git
```
You need to Install ```CMake``` and ```gcc``` for MacOS. If you don't have Homebrew, [here](https://brew.sh/) is its website.
```bash
brew install gcc
brew install cmake
```

* Build ThunderSVM. You can specify gcc as the compiler of cmake. ([path_to_g++] is typically g++-7 but may be different in your system).
```
# in thundersvm root directory
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=[path_to_g++] -DCMAKE_C_COMPILER=[path_to_gcc] -DUSE_CUDA=ON -DUSE_EIGEN=OFF ..
make -j
```

#### Installation for Windows
* Clone ThunderSVM repository
```bash
git clone git@github.com:zeyiwen/thundersvm.git
```

* Create a Visual Studio project
```bash
mkdir build
cd build
cmake .. -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE -G "Visual Studio 14 2015 Win64"
```
You need to change the Visual Studio version if you are using a different version of Visual Studio. Visual Studio can be downloaded from [this link](https://www.visualstudio.com/vs/). The above commands generate some Visual Studio project files, open the Visual Studio project to build ThunderSVM. Please note that CMake should be 3.4 or above for Windows.

#### Working without GPUs
If you don't have GPUs, ThunderSVM can work with CPU only.

* Get Eigen Library. ThunderSVM uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix calculation. To use Eigen, just 
initialize the submodule. 
```bash
# in thundersvm root directory
git submodule init eigen && git submodule update
```
* Build without GPUs for Linux
```bash
# in thundersvm root directory
mkdir build && cd build && cmake -DUSE_CUDA=OFF -DUSE_EIGEN=ON .. && make -j
```
If ```make -j``` doesn't work, please simply use ```make```. Now ThunderSVM will work solely on CPUs and does not rely on CUDA.

* Build without GPUs for MacOS
```bash
# in thundersvm root directory
mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=[path_to_g++] -DCMAKE_C_COMPILER=[path_to_gcc] -DUSE_CUDA=OFF -DUSE_EIGEN=ON .. && make -j
```

* Build without GPUs for Windows
```bash
mkdir build
cd build
cmake .. -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE -DUSE_CUDA=OFF -DUSE_EIGEN=ON -G "Visual Studio 14 2015 Win64"
```
Then, you can open the generated the Visual Studio project file to build ThunderSVM.

## Training SVMs
We show some concrete examples of using ThunderSVM. ThunderSVM uses the same command line options as LibSVM, so existing users of LibSVM can use ThunderSVM quickly. For new users of SVMs, the [parameters](parameters.md) page provides explanation for the usage of each option. 

### Training SVMs for Classification
In the following, we provide an example of using ThunderSVM for the MNIST dataset.

* Download the MNIST data set. The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2).

* Decompress the data set. For Windows machines, you can decompress the data set using tools such as [7-Zip](www.7-zip.org). For Unix based OSes, you can use
```bash
bunzip2 mnist.scale.bz2
```

* Install ThunderSVM. Instructions available in [How To](how-to.md) page.

* Run ThunderSVM
```bash
./thundersvm-train -s 0 -t 2 -g 0.125 -c 10 mnist.scale svm.model
```
The meaning of each option can be found in the [parameters](parameters.md) page. The training takes a while to complete. Once completed, you can see the classifier accuracy is 94.32%.

### Training SVMs for Regression
The usage of other SVM algorithms (such as SVM regression) are similar to the above example. The key different is the selection of the options. Let's take Abalone data set as an example.

* Download the Abalone data set. The data set is available in [this link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale).

* Install ThunderSVM. Instructions available in [How To](how-to.md) page.

* Run ThunderSVM
```bash
./thundersvm-train -s 3 -t 2 -g 3.8 -c 1000 abalone_scale svm.model
```
The meaning of each option can be found in the [parameters](parameters.md) page. 

### Interfaces
ThunderSVM provides python, R and Matlab interface. You can find the instructions in the corresponding subdirectories.
