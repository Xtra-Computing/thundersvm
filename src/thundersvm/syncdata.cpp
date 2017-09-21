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
const T *SyncData<T>::host_data() const {
    to_host();
    return static_cast<T *>(mem->host_data());
}

template<typename T>
const T *SyncData<T>::device_data() const {
    to_device();
    return static_cast<T *>(mem->device_data());
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
void SyncData<T>::to_host() const {
    mem->to_host();
}

template<typename T>
void SyncData<T>::to_device() const {
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

template<typename T>
void SyncData<T>::resize(size_t count) {
    this->mem->resize(sizeof(T) * count);
    this->count_ = count;
}

template<typename T>
void SyncData<T>::copy_from(const T *source, size_t count) {
    CUDA_CHECK(cudaMemcpy(mem->device_data(), source, sizeof(T) * count, cudaMemcpyDefault));
}

template<typename T>
void SyncData<T>::log(el::base::type::ostream_t &ostream) const {
    int i;
    ostream<<"[";
    for (i = 0; i < count()-1 && i < el::base::consts::kMaxLogPerContainer-1; ++i) {
        ostream<<host_data()[i]<<",";
    }
    ostream<<host_data()[i];
    ostream<<"]"<<std::endl;
}


template
class SyncData<int>;

template
class SyncData<real>;
