#!/bin/sh

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."


if [ ! -f "a9a" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
fi

if [ ! -f "a9a.t" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t
fi

if [ ! -f "real-sim" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2
bunzip2 real-sim.bz2
fi

if [ ! -f "mnist.scale" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
bunzip2 mnist.scale.bz2
fi

if [ ! -f "mnist.scale.t" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bunzip2 mnist.scale.t.bz2
fi

if [ ! -f "E2006.train" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2
bunzip2 E2006.train.bz2
fi

if [ ! -f "E2006.test" ]; then
wget --no-check-certificate https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2
bunzip2 E2006.test.bz2
fi
