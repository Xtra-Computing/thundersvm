#!/usr/bin/env bash
print_usage(){
    printf "usage: ./run.sh [dataset]\n"
    printf "\tsize\tclass\tfeature\n"
    printf "iris\t150\t3\t4\n"
    printf "mnist\t60000\t10\t780\n"
    printf "a9a\t32561\t2\t123\n"
    printf "a6a\t11220\t2\t123\n"
    printf "news20\t19996\t2\t1335191\n"
}
if [ $# != 1 ]
then
    print_usage
    exit
fi
DATASET_DIR=dataset
#gamma for RBF kernel
GAMMA="-g "
#penalty
C="-c "
#file name (must appear as the last argument)

case $1 in
    iris)
        GAMMA=${GAMMA}"0.5"
        C=${C}"100"
        TrainingSet=${DATASET_DIR}/"iris.scale"
        ;;
    mnist)
        GAMMA=${GAMMA}"0.125"
        C=${C}"10"
        TrainingSet=${DATASET_DIR}/"mnist.scale"
        ;;
    a9a | a6a)
        GAMMA=${GAMMA}"0.5"
        C=${C}"100"
        TrainingSet=${DATASET_DIR}/$1
        ;;
    w8a)
        GAMMA=${GAMMA}"0.5"
        C=${C}"10"
        TrainingSet=${DATASET_DIR}/"w8a"
        ;;
    news20)
        GAMMA=${GAMMA}"0.5"
        C=${C}"4"
        TrainingSet=${DATASET_DIR}/"news20.binary"
        ;;
    cov1)
        GAMMA=${GAMMA}"1"
        C=${C}"3"
        TrainingSet=${DATASET_DIR}/"cov1"
        ;;
    real-sim)
        GAMMA=${GAMMA}"0.5"
        C=${C}"4"
        TrainingSet=${DATASET_DIR}/"real-sim"
        ;;
	cifar-10)
		GAMMA=${GAMMA}"0.0025"
		C=${C}"10"
		TrainingSet=${DATASET_DIR}/"cifar10.libsvm"
		F="-f 3072"
		;;
    *)
        echo "undefined dataset, use GAMMA=0.5, C=10"
        GAMMA=${GAMMA}"0.5"
        C=${C}"10"
        TrainingSet=${DATASET_DIR}/$1
esac
###options
#svm with probability output
PROB="-b 0" #0 for no probability output; 1 for probability output.

#task type: 0 for training; 1 for cross-validation; 2 for evaluation
#	    3 for grid search; 4 for selecting better C.
TASK="-o 2"

#test set name ("e" stands for evaluation)
#In this example test set name, if you use "mnist.scale", the test set name is "mnist.scale.t"
TestSet="-e "${TrainingSet}".t"

#evaluate training error
E="-r 1" #0 not evaluate training error; evaluate training error otherwise.

#print out the command before execution
set -x

#command
./bin/release/mascot ${PROB} ${TASK} ${GAMMA} ${C} ${E} ${F} ${TestSet} ${TrainingSet}
