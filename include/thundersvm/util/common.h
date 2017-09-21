//
// Created by jiashuai on 17-9-21.
//

#ifndef THUNDERSVM_COMMON_H
#define THUNDERSVM_COMMON_H
#ifndef max

template<class T>
static inline T max(T x, T y) { return (x > y) ? x : y; }
#endif
#ifndef min
template<class T>
static inline T min(T x, T y) { return (x < y) ? x : y; }

#endif

#endif //THUNDERSVM_COMMON_H
