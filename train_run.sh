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
#./bin/thundersvm-predict ../dataset/data1 data1.model data1.predict
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

echo "run SUSY"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/SUSY
./bin/thundersvm-predict ../dataset/SUSY SUSY.model SUSY.predict
rm -rf SUSY*
fi

if [$1 -eq 7];then
echo "run url_combined_normalized"
./bin/thundersvm-train -c 100 -g 0.5 ../dataset/url_combined_normalized
./bin/thundersvm-predict ../dataset/url_combined_normalized url_combined_normalized.model url_combined_normalized.predict
rm -rf url_combined_normalized*
fi
