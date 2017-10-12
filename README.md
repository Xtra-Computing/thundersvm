# ThunderSVM
## Usage
1. clone this repository
```bash
git clone git@github.com:zeyiwen/mascot.git
```
2. download datasets
```bash
cd mascot
dataset/get_datasets.sh
```

3. build and test
```bash
mkdir build
cd build
cmake ..
make -j runtest
```
see if all tests pass

## TODO
- command line parser (3d)
- save and load models from file (3d)
- cross-validation (3-4d)
- probability estimation (1-2d)
- integrate with interfaces
## Completed
- binary/multi-class classification training/prediction
- unit test codes
- SVR
- one-class SVM
