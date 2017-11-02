//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERSVM_SYNCDATA_H
#define THUNDERSVM_SYNCDATA_H

#include "thundersvm.h"
#include "syncmem.h"

template<typename T>
class SyncData : public el::Loggable {
public:
    explicit SyncData(size_t count);

    SyncData() : mem(nullptr), size_(0) {};
    ~SyncData();

    const T *host_data() const;
    const T *device_data() const;

    T *host_data();
    T *device_data();

    void set_host_data(T *host_ptr){
        mem->set_host_data(host_ptr);
    }
    void set_device_data(T *device_ptr){
    	mem->set_device_data(device_ptr);
    }

    void to_host() const{
    	mem->to_host();
    }
    void to_device() const{
    	mem->to_device();
    }

    const T &operator[](int index) const{
    	return host_data()[index];
    }
    T &operator[](int index){
    	return host_data()[index];
    }

    void copy_from(const T *source, size_t count);
    void copy_from(const SyncData<T> &source);

    void mem_set(const T &value);

    void resize(size_t count);

    size_t mem_size() const {//number of bytes
    	return mem->size();
    }

    size_t size() const {//number of values
        return size_;
    }

    SyncMem::HEAD head() const{
    	return mem->head();
    }

    void log(el::base::type::ostream_t &ostream) const override;

private:
    SyncData<T> &operator=(const SyncData<T> &);
    SyncData(const SyncData<T>&);

    SyncMem *mem;
    size_t size_;
};

#endif //THUNDERSVM_SYNCDATA_H
