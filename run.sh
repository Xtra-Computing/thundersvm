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

#evaluate training error
E="-r 1" #0 not evaluate training error; evaluate training error otherwise.

#number of features
#NUMFEATURE="-f 16"

#file name (must appear as the last argument)
FILENAME="dataset/a9a"
#FILENAME="dataset/iris.scale"
#FILENAME="dataset/letter.scale"
#FILENAME="dataset/sector.scale"
#FILENAME="dataset/aloi.scale"
#FILENAME="dataset/glass.scale"
#FILENAME="dataset/mnist.scale"
#FILENAME="dataset/w8a"
#FILENAME="dataset/usps"
#FILENAME="dataset/shuttle.scale"
#FILENAME="dataset/cov1"


#print out the command before execution
set -x

#command
./bin/release/mascot $PROB $TASK $GAMMA $C $E $NUMFEATURE $FILENAME
