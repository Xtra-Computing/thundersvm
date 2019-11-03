#!/bin/bash

set -x

mkdir build_cputest
cd build_cputest

case ${TRAVIS_OS_NAME} in
linux)
    cmake -DBUILD_TESTS=ON -DUSE_CUDA=OFF -DUSE_EIGEN=ON ..
    ;;
osx)
   ;;
windows)
    export PATH=${MSBUILD_PATH}:$PATH
    cmake -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE -DUSE_CUDA=OFF -DUSE_EIGEN=ON -DBUILD_TESTS=ON -G "Visual Studio 14 2015 Win64" ..
    ;;
esac

cmake --build . --target runtest
