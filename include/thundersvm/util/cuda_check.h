//
// Created by jiashuai on 17-9-16.
//

#ifndef THUNDERSVM_CUDA_CHECK_H
#define THUNDERSVM_CUDA_CHECK_H
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#endif //THUNDERSVM_CUDA_CHECK_H
