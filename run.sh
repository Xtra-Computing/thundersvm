#!/usr/bin/env bash


###options
#svm with probability output
PROB="-b 0" #0 for no probability output; 1 for probability output.

#task type
TASK="-o 0" #0 for training; 1 for cross validation; 2 for evaluation

#gamma for RBF kernel
GAMMA="-g 0.382"

#penalty
C="-c 100"

#number of features
NUMFEATURE="-f 123"

#file name (must appear as the last argument)
FILENAME="dataset/a1a" #"dataset/iris.scale"

#print out the command before execution
set -x

#command
./bin/mascot $PROB $TASK $GAMMA $C $NUMFEATURE $FILENAME
