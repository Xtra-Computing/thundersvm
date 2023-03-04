#!/bin/bash
make clean
make -j 10
if [ $1 -eq 0 ];then

echo "run test data"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/test_dataset.txt
./bin/thundersvm-predict ../dataset/test_dataset.txt test_dataset.txt.model test_dataset.predict
rm -rf test_dataset.*
fi


if [ $1 -eq 1 ];then

echo "run data1"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/data1
./bin/thundersvm-predict ../dataset/data1 data1.model data1.predict
rm -rf data1*
fi

if [ $1 -eq 2 ];then

echo "run a9a"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/a9a
./bin/thundersvm-predict ../dataset/a9a a9a.model a9a.predict
rm -rf a9a*
fi


if [ $1 -eq 3 ];then

echo "run mnist"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/mnist.scale
./bin/thundersvm-predict ../dataset/mnist.scale mnist.scale.model mnist.scale.predict
rm -rf mnist*
fi
