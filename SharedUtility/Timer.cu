//
// Created by ss on 17-1-18.
//
#include "Timer.h"

float Timer::diff(timeval t1, timeval t2) {
    return (t2.tv_usec - t1.tv_usec) + (t2.tv_sec - t1.tv_sec) * 1000000;
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
    count++;
}

float Timer::getTotalTime() {
    return accumulator / 1e6f;
}

float Timer::getAverageTime() {
    return getTotalTime() / getCount();
}

unsigned int Timer::getCount() {
    return count;
}

Timer trainingTimer;
Timer calculateKernelTimer;
Timer preComputeTimer;
Timer selectTimer;
Timer updateAlphaTimer;
Timer updateGTimer;
Timer iterationTimer;
Timer initTimer;

