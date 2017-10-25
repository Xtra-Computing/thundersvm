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

inline int max2power(int n) {
    return int(pow(2, floor(log2f(float(n)))));
}
const int BLOCK_SIZE = 512;

//inline int GET_BLOCKS(const int n) {
//    return (n - 1) / BLOCK_SIZE + 1;
//}
const int NUM_BLOCKS = 32 * 56;

#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#endif //THUNDERSVM_COMMON_H
