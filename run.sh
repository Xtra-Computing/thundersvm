#!/usr/bin/env bash


###options
#svm with probability output
PROB="-b 0" #0 for no probability output; 1 for probability output.

#task type
TASK="-o 2" #0 for training; 1 for cross validation; 2 for evaluation

#gamma for RBF kernel
GAMMA="-g 0.125"

#penalty
C="-c 10"

#number of features
#NUMFEATURE="-f 16"

#file name (must appear as the last argument)
FILENAME="dataset/a9a" #"dataset/iris.scale"
#FILENAME="dataset/iris.scale" #"dataset/iris.scale"
#FILENAME="dataset/letter.scale" #"dataset/iris.scale"
#FILENAME="dataset/sector.scale" #"dataset/iris.scale"
#FILENAME="dataset/aloi.scale" #"dataset/iris.scale"
#FILENAME="dataset/glass.scale" #"dataset/iris.scale"
FILENAME="dataset/mnist.scale" #"dataset/iris.scale"
#FILENAME="dataset/w8a" #"dataset/iris.scale"
#FILENAME="dataset/usps" #"dataset/iris.scale"
#FILENAME="dataset/shuttle.scale" #"dataset/iris.scale"
#FILENAME="dataset/cov1" #"dataset/iris.scale"


#print out the command before execution
set -x

#command
./bin/mascot $PROB $TASK $GAMMA $C $NUMFEATURE $FILENAME
