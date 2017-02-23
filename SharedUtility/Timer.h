//
// Created by ss on 17-1-18.
//

#ifndef MASCOT_SVM_TIMER_H
#define MASCOT_SVM_TIMER_H
/**
 * Utility for counting time.
 *
 * Usage:
 * 1. add declaration in this file
 *      in header file
 *      extern Timer timer;
 *      in source file
 *      Timer timer;
 * 2. add timer code
 *      TIMER_START(timer)
 *      **code**
 *      TIMER_STOP(timer)
 *    or
 *      ACCUMULATE_TIME(timer, **code**)
 * 3. print timer
 *      PRINT_TIME(**message**, timer)
 */
#include <cstdlib>
#include "sys/time.h"
#include <vector>
using std::vector;
#define COUNT_TIME
class Timer {
    float diff(timeval t1, timeval t2);

    void getTime(timeval &t);

    float accumulator;
    unsigned int count;
    timeval begin;
    timeval end;
public:
    Timer():accumulator(0) {};
    void start();

    void stop();

    float getTotalTime();
    float getAverageTime();

    unsigned int  getCount();
};
extern Timer trainingTimer;
extern Timer calculateKernelTimer;
extern Timer preComputeTimer;
extern Timer selectTimer;
extern Timer updateAlphaTimer;
extern Timer updateGTimer;
extern Timer iterationTimer;
extern Timer initTimer;
#ifdef COUNT_TIME
#define TIMER_START(timer) cudaDeviceSynchronize();timer.start();
#define TIMER_STOP(timer) cudaDeviceSynchronize();timer.stop();
#define ACCUMULATE_TIME(timer, code) TIMER_START(timer);code;TIMER_STOP(timer);

#define PRINT_TIME(msg, timer) printf("%s time : %fs, count : %d, ave : %f \n", msg, timer.getTotalTime(),timer.getCount(),timer.getAverageTime());
#else
#define TIMER_START(timer)
#define TIMER_STOP(timer)
#define PRINT_TIME(string,timer)
#define ACCUMULATE_TIME(timer, code) code;
#endif

#endif //MASCOT_SVM_TIMER_H
