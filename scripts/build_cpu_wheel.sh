#!/bin/bash

set -x

mkdir build
cd build

case ${TRAVIS_OS_NAME} in
linux)
    cmake -DUSE_CUDA=OFF -DUSE_EIGEN=ON ..
    ;;
osx)
   ;;
windows)
    export PATH=${MSBUILD_PATH}:$PATH
    cmake -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE  -DUSE_CUDA=OFF -DUSE_EIGEN=ON -G "Visual Studio 14 2015 Win64" ..
    choco install python --version=3.6.3
    python -m pip install --upgrade pip
    pip install wheel
    ;;
esac

cmake --build .
cd ../python
python setup.py bdist_wheel
