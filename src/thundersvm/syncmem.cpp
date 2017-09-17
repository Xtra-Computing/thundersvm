//
// Created by jiashuai on 17-9-16.
//

#include <thundersvm/syncmem.h>

SyncMem::SyncMem() : gpu_ptr(nullptr), cpu_ptr(nullptr), size_(0), head_(UNINITIALIZED) {

}

SyncMem::SyncMem(size_t size) : gpu_ptr(nullptr), cpu_ptr(nullptr), size_(size), head_(UNINITIALIZED) {

}

SyncMem::~SyncMem() {
    if (cpu_ptr)
        CUDA_CHECK(cudaFreeHost(cpu_ptr));
    if (gpu_ptr)
        CUDA_CHECK(cudaFree(gpu_ptr));
}

void *SyncMem::cpu_data() {
    switch (head_) {
        case UNINITIALIZED:
            CUDA_CHECK(cudaMallocHost(&cpu_ptr, size_));
            head_ = CPU;
            return cpu_ptr;
        case GPU:
            if (nullptr == cpu_ptr)
                CUDA_CHECK(cudaMallocHost(&cpu_ptr, size_));
            CUDA_CHECK(cudaMemcpy(cpu_ptr, gpu_ptr, size_, cudaMemcpyDeviceToHost));
            head_ = CPU;
            return cpu_ptr;
        case CPU:
            return cpu_ptr;
    }
}

void *SyncMem::gpu_data() {
    switch (head_) {
        case UNINITIALIZED:
            CUDA_CHECK(cudaMalloc(&gpu_ptr, size_));
            head_ = GPU;
            return gpu_ptr;
        case CPU:
            if (nullptr == gpu_ptr)
                CUDA_CHECK(cudaMalloc(&gpu_ptr, size_));
            CUDA_CHECK(cudaMemcpy(gpu_ptr, cpu_ptr, size_, cudaMemcpyHostToDevice));
            head_ = GPU;
            return gpu_ptr;
        case GPU:
            return gpu_ptr;
    }
}

size_t SyncMem::size() const {
    return size_;
}

SyncMem::HEAD SyncMem::head() const {
    return head_;
}
