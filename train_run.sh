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

echo "run a1a"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/a1a
./bin/thundersvm-predict ../dataset/a1a a1a.model a1a.predict
rm -rf a1a*
fi


if [ $1 -eq 3 ];then

echo "run a9a"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/a9a
./bin/thundersvm-predict ../dataset/a9a a9a.model a9a.predict
rm -rf a9a*
fi

if [ $1 -eq 4 ];then

echo "run real-sim"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/real-sim
./bin/thundersvm-predict ../dataset/real-sim real-sim.model real-sim.predict
rm -rf real-sim*
fi

if [ $1 -eq 5 ];then

echo "run w8a"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/w8a
./bin/thundersvm-predict ../dataset/w8a w8a.model w8a.predict
rm -rf w8a*
fi

if [ $1 -eq 6 ];then

echo "run rcv1_test.binary"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/rcv1_test.binary
./bin/thundersvm-predict ../dataset/rcv1_test.binary rcv1_test.binary.model rcv1_test.binary.predict
rm -rf rcv1_test.binary*
fi

if [ $1 -eq 7 ];then
echo "run url_combined_normalized"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/url_combined_normalized
./bin/thundersvm-predict ../dataset/url_combined_normalized url_combined_normalized.model url_combined_normalized.predict
rm -rf url_combined_normalized*
fi

if [ $1 -eq 8 ];then
echo "run epsilon_normalized"
./bin/thundersvm-train -c 0.01 -g 1 ../dataset/epsilon_normalized
./bin/thundersvm-predict ../dataset/epsilon_normalized epsilon_normalized.model epsilon_normalized.predict
rm -rf epsilon_normalized*
fi

if [ $1 -eq 9 ];then
echo "run E2006.train"
./bin/thundersvm-train -s 3 -c 256 -g 0.125 ../dataset/E2006.train
./bin/thundersvm-predict ../dataset/E2006.train E2006.train.model E2006.train.predict
rm -rf E2006.train*
fi

if [ $1 -eq 10 ];then
echo "run mnist.scale"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/mnist.scale
./bin/thundersvm-predict ../dataset/mnist.scale mnist.scale.model mnist.scale.predict
rm -rf mnist.scale*
fi

if [ $1 -eq 11 ];then
echo "run gisette_scale"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/gisette_scale
./bin/thundersvm-predict ../dataset/gisette_scale gisette_scale.model gisette_scale.predict
rm -rf gisette_scale*
fi

if [ $1 -eq 12 ];then
echo "run ijcnn1.t"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/ijcnn1.t
./bin/thundersvm-predict ../dataset/ijcnn1.t ijcnn1.t.model ijcnn1.t.predict
rm -rf ijcnn1.t*
fi


