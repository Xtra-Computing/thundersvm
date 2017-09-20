//
// Created by jiashuai on 17-9-16.
//

#include <thundersvm/syncmem.h>

SyncMem::SyncMem() : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED) {

}

SyncMem::SyncMem(size_t size) : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED) {

}

SyncMem::~SyncMem() {
    this->head_ = UNINITIALIZED;
    if (host_ptr) {
        CUDA_CHECK(cudaFreeHost(host_ptr));
        host_ptr = nullptr;
    }
    if (device_ptr) {
        CUDA_CHECK(cudaFree(device_ptr));
        device_ptr = nullptr;
    }
}

void *SyncMem::host_data() {
    to_host();
    return host_ptr;
}

void *SyncMem::device_data() {
    to_device();
    return device_ptr;
}

size_t SyncMem::size() const {
    return size_;
}

SyncMem::HEAD SyncMem::head() const {
    return head_;
}

void SyncMem::to_host() {
    switch (head_) {
        case UNINITIALIZED:
            CUDA_CHECK(cudaMallocHost(&host_ptr, size_));
            head_ = HOST;
            break;
        case DEVICE:
            if (nullptr == host_ptr)
                CUDA_CHECK(cudaMallocHost(&host_ptr, size_));
            CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size_, cudaMemcpyDeviceToHost));
            head_ = HOST;
            break;
        case HOST:;
    }
}

void SyncMem::to_device() {
    switch (head_) {
        case UNINITIALIZED:
            CUDA_CHECK(cudaMalloc(&device_ptr, size_));
            head_ = DEVICE;
            break;
        case HOST:
            if (nullptr == device_ptr)
                CUDA_CHECK(cudaMalloc(&device_ptr, size_));
            CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size_, cudaMemcpyHostToDevice));
            head_ = DEVICE;
            break;
        case DEVICE:;
    }
}

void SyncMem::resize(size_t size) {
    this->~SyncMem();
    this->size_ = size;
}
