//
// Created by jiashuai on 17-9-17.
//
#include "thundersvm/syncdata.h"

template<typename T>
SyncData<T>::SyncData(size_t count):mem(new SyncMem(sizeof(T) * count)), size_(count) {

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
void SyncData<T>::resize(size_t count) {
    delete mem;
    mem = new SyncMem(sizeof(T) * count);
    this->size_ = count;
}

template<typename T>
void SyncData<T>::copy_from(const T *source, size_t count) {
    CUDA_CHECK(cudaMemcpy(mem->device_data(), source, sizeof(T) * count, cudaMemcpyDefault));
    //CUDA_CHECK(cudaMemcpy(mem->host_data(), source, sizeof(T) * count, cudaMemcpyDefault));
}


template<typename T>
void SyncData<T>::log(el::base::type::ostream_t &ostream) const {
    int i;
    ostream << "[";
    for (i = 0; i < size() - 1 && i < el::base::consts::kMaxLogPerContainer - 1; ++i) {
        ostream << host_data()[i] << ",";
    }
    ostream << host_data()[i];
    ostream << "]";
}

template<typename T>
void SyncData<T>::copy_from(const SyncData<T> &source) {
    CHECK_EQ(size(), source.size()) << "destination and source count doesn't match";
    copy_from(source.device_data(), source.size());
//    copy_from(source.host_data(), source.size());
}

template<typename T>
void SyncData<T>::mem_set(const T &value) {
    CUDA_CHECK(cudaMemset(device_data(), value, mem_size()));
}

template
class SyncData<int>;

template
class SyncData<real>;
