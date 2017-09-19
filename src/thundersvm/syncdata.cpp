//
// Created by jiashuai on 17-9-17.
//
#include "thundersvm/syncdata.h"

template<typename T>
SyncData<T>::SyncData(size_t count):mem(new SyncMem(sizeof(T) * count)), count_(count) {

}

template<typename T>
SyncData<T>::~SyncData() {
    delete mem;
}

template<typename T>
T *SyncData<T>::data() {
    CHECK_NE(head(), SyncMem::UNINITIALIZED);
    switch (mem->head()) {
        case SyncMem::HOST:
            return host_data();
        case SyncMem::DEVICE:
            return device_data();
            //should never reach this
        case SyncMem::UNINITIALIZED:
            break;
    }
}

template<typename T>
T *SyncData<T>::host_data() {
    to_host();
    return static_cast<T *>(mem->host_data());
}

template<typename T>
T *SyncData<T>::device_data() {
    to_device();
    return static_cast<T *>(mem->device_data());
}

template<typename T>
void SyncData<T>::to_host() {
    mem->to_host();
}

template<typename T>
void SyncData<T>::to_device() {
    mem->to_device();
}

template<typename T>
size_t SyncData<T>::size() const {
    return mem->size();
}

template<typename T>
SyncMem::HEAD SyncData<T>::head() const {
    return mem->head();
}

template<typename T>
size_t SyncData<T>::count() const {
    return count_;
}

template
class SyncData<int>;

template
class SyncData<real>;
