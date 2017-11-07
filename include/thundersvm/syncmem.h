//
// Created by jiashuai on 17-9-16.
//

#ifndef THUNDERSVM_SYNCMEM_H
#define THUNDERSVM_SYNCMEM_H

#include <thundersvm/thundersvm.h>

namespace thunder {
    inline void malloc_host(void **ptr, size_t size) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMallocHost(ptr, size));
#else
        *ptr = malloc(size);
#endif
    }

    inline void free_host(void *ptr) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaFreeHost(ptr));
#else
        free(ptr);
#endif
    }

    inline void device_mem_copy(void *dst, const void *src, size_t size) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
#else
        NO_GPU;
#endif
    }

    class SyncMem {
    public:
        SyncMem();

        explicit SyncMem(size_t size);

        ~SyncMem();

        void *host_data();

        void *device_data();

        void set_host_data(void *data);

        void set_device_data(void *data);

        void to_host();

        void to_device();

        size_t size() const;

        enum HEAD {
            HOST, DEVICE, UNINITIALIZED
        };

        HEAD head() const;

    private:
        void *device_ptr;
        void *host_ptr;
        bool own_device_data;
        bool own_host_data;
        size_t size_;
        HEAD head_;
    };
}
using thunder::SyncMem;
#endif //THUNDERSVM_SYNCMEM_H
