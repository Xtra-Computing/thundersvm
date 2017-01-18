//
// Created by ss on 17-1-18.
//
#include "Timer.h"

float Timer::diff(timeval t1, timeval t2) {
    return (t2.tv_usec - t1.tv_usec) / 1e6f + (t2.tv_sec - t1.tv_sec);
}

void Timer::getTime(timeval &t) {
    gettimeofday(&t, NULL);
}

void Timer::start() {
    getTime(begin);
}

void Timer::stop() {
    getTime(end);
    accumulator += diff(begin, end);
}

float Timer::getTotalTime() {
    return accumulator;
}
Timer trainingTimer;
Timer calculateKernelTimer;
Timer preComputeTimer;
Timer selectTimer;
Timer updateAlphaTimer;
Timer updateGTimer;
Timer iterationTimer;

