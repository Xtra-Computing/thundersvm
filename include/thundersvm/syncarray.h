//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERSVM_SYNCDATA_H
#define THUNDERSVM_SYNCDATA_H

#include "thundersvm.h"
#include "syncmem.h"

/**
 * @brief Wrapper of SyncMem with a type
 * @tparam T type of element
 */
template<typename T>
class SyncArray : public el::Loggable {
public:
    /**
     * initialize class that can store given count of elements
     * @param count the given count
     */
    explicit SyncArray(size_t count);

    SyncArray() : mem(nullptr), size_(0) {};

    ~SyncArray();

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

    //deprecated because of performance issue
//    /**
//     * random access operator
//     * @param index the index of the elements
//     * @return **host** element at the index
//     */
//    const T &operator[](int index) const{
//        return host_data()[index];
//    }
//
//    T &operator[](int index){
//        return host_data()[index];
//    }

    /**
     * copy device data. This will call to_device() implicitly.
     * @param source source device data pointer
     * @param count the count of elements
     */
    void copy_from(const T *source, size_t count);

    void copy_from(const SyncArray<T> &source);

    /**
     * set all elements to the given value. This method will set device data.
     * @param value
     */
    void mem_set(const T &value);

    /**
     * resize to a new size. This will also clear all data.
     * @param count
     */
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
    SyncArray<T> &operator=(const SyncArray<T> &);

    SyncArray(const SyncArray<T> &);

    SyncMem *mem;
    size_t size_;
};

#endif //THUNDERSVM_SYNCDATA_H
