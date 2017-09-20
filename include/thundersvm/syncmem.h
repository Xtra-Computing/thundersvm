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
    void* host_data();
    void* device_data();
    void to_host();
    void to_device();
    size_t size() const;
    void resize(size_t size) ;
    enum HEAD {HOST, DEVICE, UNINITIALIZED};
    HEAD head() const;

private:
    void *device_ptr;
    void *host_ptr;
    size_t size_;
    HEAD head_;
};
#endif //THUNDERSVM_SYNCMEM_H
