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
    SyncData() : mem(nullptr), count_(0) {};

    explicit SyncData(size_t count);

    ~SyncData();

    const T *host_data() const;

    void set_host_data(T *host_ptr);

    T *host_data();

    const T *device_data() const;

    void set_device_data(T *device_ptr);

    T *device_data();

    void to_host() const;

    void to_device() const;

    T &operator[](int index);

    const T &operator[](int index) const;

    void copy_from(const T *source, size_t count);

    void copy_from(const SyncData<T> &source);

    void mem_set(const T &value);

    size_t size() const;

    void resize(size_t count);

    size_t count() const;

    SyncMem::HEAD head() const;

    void log(el::base::type::ostream_t &ostream) const override;

private:
    SyncData<T> &operator=(const SyncData<T> &);
    SyncData(const SyncData<T>&);

    SyncMem *mem;
    size_t count_;
};

#endif //THUNDERSVM_SYNCDATA_H
