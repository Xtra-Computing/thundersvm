//
// Created by jiashuai on 17-9-16.
//

#ifndef THUNDERSVM_SYNCMEM_H
#define THUNDERSVM_SYNCMEM_H

#include "thundersvm.h"
class SyncMem{
public:
    SyncMem();
    explicit SyncMem(size_t size);
    ~SyncMem();
    void* cpu_data();
    void* gpu_data();
    size_t size() const;
    enum HEAD {CPU, GPU, UNINITIALIZED};
    HEAD head() const;
private:
    void *gpu_ptr;
    void *cpu_ptr;
    size_t size_;
    HEAD head_;
};
#endif //THUNDERSVM_SYNCMEM_H
