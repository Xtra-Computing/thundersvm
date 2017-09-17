//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERSVM_SYNCDATA_H
#define THUNDERSVM_SYNCDATA_H

#include "thundersvm.h"
#include "syncmem.h"
template <typename T>
class SyncData{
public:
    SyncData():mem(nullptr), count_(0){};
    explicit SyncData(size_t count);
    ~SyncData();
    T *data();
    T *host_data();
    T *device_data();
    void to_host();
    void to_device();
    size_t size() const;
    size_t count() const;
    SyncMem::HEAD head() const;
private:
    SyncMem *mem;
    size_t count_;
};
#endif //THUNDERSVM_SYNCDATA_H
