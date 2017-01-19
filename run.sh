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
        FILENAME=${DATASET_DIR}/"iris.scale"
        ;;
    mnist)
        GAMMA=${GAMMA}"0.125"
        C=${C}"10"
        FILENAME=${DATASET_DIR}/"mnist.scale"
        ;;
    a9a | a6a)
        GAMMA=${GAMMA}"0.5"
        C=${C}"100"
        FILENAME=${DATASET_DIR}/$1
        ;;
    w8a)
        GAMMA=${GAMMA}"0.5"
        C=${C}"10"
        FILENAME=${DATASET_DIR}/"w8a"
        ;;
    news20)
        GAMMA=${GAMMA}"0.5"
        C=${C}"4"
        FILENAME=${DATASET_DIR}/"news20.binary"
        ;;
    *)
        echo "undefined dataset, use GAMMA=0.5, C=10"
        GAMMA=${GAMMA}"0.5"
        C=${C}"10"
        FILENAME=${DATASET_DIR}/$1
esac
###options
#svm with probability output
PROB="-b 0" #0 for no probability output; 1 for probability output.

#task type
TASK="-o 2" #0, 1, 2 and 3 for training, cross-validation, evaluation and grid search, respectively.

#evaluate training error
E="-r 1" #0 not evaluate training error; evaluate training error otherwise.

#number of features
#NUMFEATURE="-f 16"

#print out the command before execution
set -x

#command
./bin/release/mascot ${PROB} ${TASK} ${GAMMA} ${C} ${E} ${NUMFEATURE} ${FILENAME}
