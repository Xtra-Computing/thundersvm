//
// Created by jiashuai on 17-9-16.
//

#include <thundersvm/syncmem.h>

namespace thunder {

    void SyncMem::malloc_host(void **ptr, size_t size) {
#ifdef USE_CUDA
        host_allocator.DeviceAllocate(ptr, size);
#else
        *ptr = new char[size];
#endif
    }

    void SyncMem::free_host(void *ptr) {
#ifdef USE_CUDA
        host_allocator.DeviceFree(ptr);
#else
        free(ptr);
#endif
    }

    DeviceAllocator SyncMem::device_allocator(2, 3, 11, CachingDeviceAllocator::INVALID_SIZE, true, false);
    HostAllocator SyncMem::host_allocator(2, 3, 11, CachingDeviceAllocator::INVALID_SIZE, false, false);


    size_t SyncMem::total_memory_size = 0;
    SyncMem::SyncMem() : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED), own_device_data(false),
                         own_host_data(false) {

    }

    SyncMem::SyncMem(size_t size) : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED),
                                    own_device_data(false), own_host_data(false) {

    }

    SyncMem::~SyncMem() {
        // if (this->head_ != UNINITIALIZED) {
        this->head_ = UNINITIALIZED;
        if (own_host_data || own_device_data) total_memory_size -= size_;
        if (host_ptr && own_host_data) {
            free_host(host_ptr);
            host_ptr = nullptr;
        }
#ifdef USE_CUDA
            // if (device_ptr && own_device_data) {
            //     CUDA_CHECK(cudaFree(device_ptr));
            //     device_ptr = nullptr;
            // }
        if (device_ptr && own_device_data) {
            device_allocator.DeviceFree(device_ptr);
            device_ptr = nullptr;
        }
#endif
        // }
    }

    void *SyncMem::host_data() {
        to_host();
        return host_ptr;
    }

    void *SyncMem::device_data() {
#ifdef USE_CUDA
        to_device();
#else
        NO_GPU;
#endif
        return device_ptr;
    }

    size_t SyncMem::size() const {
        return size_;
    }

    SyncMem::HEAD SyncMem::head() const {
        return head_;
    }

    void SyncMem::to_host() {
        switch (head_) {
            case UNINITIALIZED:
                //std::cout<<"UNINITIALIZED to_host operation"<<std::endl;
                malloc_host(&host_ptr, size_);
                // memset(host_ptr, 0, size_);
                CUDA_CHECK(cudaMemset(host_ptr, 0, size_));
                head_ = HOST;
                own_host_data = true;
                total_memory_size += size_;
                break;
            case DEVICE:
#ifdef USE_CUDA
                if (nullptr == host_ptr) {
                    // CUDA_CHECK(cudaMallocHost(&host_ptr, size_));
                    // CUDA_CHECK(cudaMemset(host_ptr, 0, size_));
                    // own_host_data = true;
                    //std::cout<<"device to_host operation"<<std::endl;
                    malloc_host(&host_ptr, size_);
                    CUDA_CHECK(cudaMemset(host_ptr, 0, size_));
                    head_ = HOST;
                    own_host_data = true;
                }
                CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size_, cudaMemcpyDeviceToHost));
                head_ = HOST;
#else
                NO_GPU;
#endif
                break;
            case HOST:;
        }
    }

    void SyncMem::to_device() {
#ifdef USE_CUDA
        switch (head_) {
            case UNINITIALIZED:
                //std::cout<<"UNINITIALIZED to_device operation"<<std::endl;
                // CUDA_CHECK(cudaMalloc(&device_ptr, size_));
                CUDA_CHECK(device_allocator.DeviceAllocate(&device_ptr, size_));
                CUDA_CHECK(cudaMemset(device_ptr, 0, size_));
                head_ = DEVICE;
                own_device_data = true;
                total_memory_size += size_;
                break;
            case HOST:
                if (nullptr == device_ptr) {
                    //std::cout<<"host to_device operation"<<std::endl;
                    // CUDA_CHECK(cudaMalloc(&device_ptr, size_));
                    CUDA_CHECK(device_allocator.DeviceAllocate(&device_ptr, size_));
                    CUDA_CHECK(cudaMemset(device_ptr, 0, size_));
                    own_device_data = true;
                }
                CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size_, cudaMemcpyHostToDevice));
                head_ = DEVICE;
                break;
            case DEVICE:;
        }
#else
        NO_GPU;
#endif
    }

    void SyncMem::set_host_data(void *data) {
        CHECK_NOTNULL(data);
        if (own_host_data) {
            free_host(host_ptr);
            total_memory_size -= size_;
        }
        host_ptr = data;
        own_host_data = false;
        head_ = HEAD::HOST;
    }

    void SyncMem::set_device_data(void *data) {
#ifdef USE_CUDA
        CHECK_NOTNULL(data);
        if (own_device_data) {
            // CUDA_CHECK(cudaFree(device_data()));
            device_allocator.DeviceFree(device_ptr);
            total_memory_size -= size_;
        }
        device_ptr = data;
        own_device_data = false;
        head_ = HEAD::DEVICE;
#else
        NO_GPU;
#endif
    }


    //CachingDeviceAllocator 

    cudaError_t DeviceAllocator::DeviceAllocate(int device, void **d_ptr, size_t bytes, cudaStream_t active_stream) {
        *d_ptr = NULL;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
            if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
            device = entrypoint_device;
        }

        // Create a block descriptor for the requested allocation
        bool found = false;
        BlockDescriptor search_key(device);
        search_key.associated_stream = active_stream;
        NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

        if (search_key.bin > max_bin) {
            // Bin is greater than our maximum bin: allocate the request
            // exactly and give out-of-bounds bin.  It will not be cached
            // for reuse when returned.
            search_key.bin = max_bin + 1;
            search_key.bytes = bytes;
        } else {
            // Search for a suitable cached allocation: lock

            if (search_key.bin < min_bin) {
                // Bin is less than minimum bin: round up
                search_key.bin = min_bin;
                search_key.bytes = min_bin_bytes;
            }
        }

        mutex.Lock();
        // Iterate through the range of cached blocks on the same device in the same bin
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
        while ((block_itr != cached_blocks.end())
               && (block_itr->device == device)
               && (block_itr->bin == search_key.bin)
               && (block_itr->bytes == search_key.bytes)) {
            // To prevent races with reusing blocks returned by the host but still
            // in use by the device, only consider cached blocks that are
            // either (from the active stream) or (from an idle stream)
            if ((active_stream == block_itr->associated_stream) ||
                (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady)) {
                // Reuse existing cache block.  Insert into live blocks.
                found = true;
                search_key = *block_itr;
                search_key.associated_stream = active_stream;
                live_blocks.insert(search_key);

                // Remove from free blocks
//                cached_bytes[device].free -= search_key.bytes;
//                cached_bytes[device].live += search_key.bytes;
                cached_bytes[device].free -= block_itr->bytes;
                cached_bytes[device].live += block_itr->bytes;

                if (debug)
                    _CubLog("\tDevice %d reused cached block at %p (%lld bytes) for stream %lld (previously associated with stream %lld).\n",
                            device, search_key.d_ptr, (long long) search_key.bytes,
                            (long long) search_key.associated_stream, (long long) block_itr->associated_stream);

                cached_blocks.erase(block_itr);

                break;
            }
            block_itr++;
        }

        // Done searching: unlock
        mutex.Unlock();

        // Allocate the block if necessary
        if (!found) {
            // Set runtime's current device to specified device (entrypoint may not be set)
            if (device != entrypoint_device) {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
                if (CubDebug(error = cudaSetDevice(device))) return error;
            }

            // Attempt to allocate
            if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes)) == cudaErrorMemoryAllocation) {
                // The allocation attempt failed: free all cached blocks on device and retry
                if (debug)
                    _CubLog("\tDevice %d failed to allocate %lld bytes for stream %lld, retrying after freeing cached allocations",
                            device, (long long) search_key.bytes, (long long) search_key.associated_stream);

                error = cudaSuccess;    // Reset the error we will return
                cudaGetLastError();     // Reset CUDART's error

                // Lock
                mutex.Lock();

                // Iterate the range of free blocks on the same device
                BlockDescriptor free_key(device);
                CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

                while ((block_itr != cached_blocks.end()) && (block_itr->device == device)) {
                    // No need to worry about synchronization with the device: cudaFree is
                    // blocking and will synchronize across all kernels executing
                    // on the current device

                    // Free device memory and destroy stream event.
                    if (CubDebug(error = cudaFree(block_itr->d_ptr))) break;
                    if (CubDebug(error = cudaEventDestroy(block_itr->ready_event))) break;

                    // Reduce balance and erase entry
                    cached_bytes[device].free -= block_itr->bytes;

                    if (debug)
                        _CubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                                device, (long long) block_itr->bytes, (long long) cached_blocks.size(),
                                (long long) cached_bytes[device].free, (long long) live_blocks.size(),
                                (long long) cached_bytes[device].live);

                    cached_blocks.erase(block_itr);

                    block_itr++;
                }

                // Unlock
                mutex.Unlock();

                // Return under error
                if (error) return error;

                // Try to allocate again
                if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) return error;
            }

            // Create ready event
            if (CubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
                return error;

            // Insert into live blocks
            mutex.Lock();
            live_blocks.insert(search_key);
            cached_bytes[device].live += search_key.bytes;
            mutex.Unlock();

            if (debug)
                _CubLog("\tDevice %d allocated new device block at %p (%lld bytes associated with stream %lld).\n",
                        device, search_key.d_ptr, (long long) search_key.bytes,
                        (long long) search_key.associated_stream);

            // Attempt to revert back to previous device if necessary
            if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device)) {
                if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
            }
        }

        // Copy device pointer to output parameter
        *d_ptr = search_key.d_ptr;

        if (debug)
            _CubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
                    (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
                    (long long) live_blocks.size(), (long long) cached_bytes[device].live);

        return error;
    }

    cudaError_t DeviceAllocator::DeviceAllocate(void **d_ptr, size_t bytes, cudaStream_t active_stream) {
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    }

    cudaError_t HostAllocator::DeviceAllocate(int device, void **d_ptr, size_t bytes, cudaStream_t active_stream) {
        *d_ptr = NULL;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
            if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
            device = entrypoint_device;
        }

        // Create a block descriptor for the requested allocation
        bool found = false;
        BlockDescriptor search_key(device);
        search_key.associated_stream = active_stream;
        NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

        if (search_key.bin > max_bin) {
            // Bin is greater than our maximum bin: allocate the request
            // exactly and give out-of-bounds bin.  It will not be cached
            // for reuse when returned.
            search_key.bin = max_bin + 1;
            search_key.bytes = bytes;
        } else {
            // Search for a suitable cached allocation: lock

            if (search_key.bin < min_bin) {
                // Bin is less than minimum bin: round up
                search_key.bin = min_bin;
                search_key.bytes = min_bin_bytes;
            }
        }

        mutex.Lock();
        // Iterate through the range of cached blocks on the same device in the same bin
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
        while ((block_itr != cached_blocks.end())
               && (block_itr->device == device)
               && (block_itr->bin == search_key.bin)
               && (block_itr->bytes == search_key.bytes)) {
            // To prevent races with reusing blocks returned by the host but still
            // in use by the device, only consider cached blocks that are
            // either (from the active stream) or (from an idle stream)
            if ((active_stream == block_itr->associated_stream) ||
                (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady)) {
                // Reuse existing cache block.  Insert into live blocks.
                found = true;
                search_key = *block_itr;
                search_key.associated_stream = active_stream;
                live_blocks.insert(search_key);

                // Remove from free blocks
//                cached_bytes[device].free -= search_key.bytes;
//                cached_bytes[device].live += search_key.bytes;
                cached_bytes[device].free -= block_itr->bytes;
                cached_bytes[device].live += block_itr->bytes;

                if (debug)
                    _CubLog("\tDevice %d reused cached block at %p (%lld bytes) for stream %lld (previously associated with stream %lld).\n",
                            device, search_key.d_ptr, (long long) search_key.bytes,
                            (long long) search_key.associated_stream, (long long) block_itr->associated_stream);

                cached_blocks.erase(block_itr);

                break;
            }
            block_itr++;
        }

        // Done searching: unlock
        mutex.Unlock();

        // Allocate the block if necessary
        if (!found) {
            // Set runtime's current device to specified device (entrypoint may not be set)
            if (device != entrypoint_device) {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
                if (CubDebug(error = cudaSetDevice(device))) return error;
            }

            // Attempt to allocate
            if (CubDebug(error = cudaMallocHost(&search_key.d_ptr, search_key.bytes)) == cudaErrorMemoryAllocation) {
                // The allocation attempt failed: free all cached blocks on device and retry
                if (debug)
                    _CubLog("\tDevice %d failed to allocate %lld bytes for stream %lld, retrying after freeing cached allocations",
                            device, (long long) search_key.bytes, (long long) search_key.associated_stream);

                error = cudaSuccess;    // Reset the error we will return
                cudaGetLastError();     // Reset CUDART's error

                // Lock
                mutex.Lock();

                // Iterate the range of free blocks on the same device
                BlockDescriptor free_key(device);
                CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

                while ((block_itr != cached_blocks.end()) && (block_itr->device == device)) {
                    // No need to worry about synchronization with the device: cudaFree is
                    // blocking and will synchronize across all kernels executing
                    // on the current device

                    // Free device memory and destroy stream event.
                    if (CubDebug(error = cudaFree(block_itr->d_ptr))) break;
                    if (CubDebug(error = cudaEventDestroy(block_itr->ready_event))) break;

                    // Reduce balance and erase entry
                    cached_bytes[device].free -= block_itr->bytes;

                    if (debug)
                        _CubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                                device, (long long) block_itr->bytes, (long long) cached_blocks.size(),
                                (long long) cached_bytes[device].free, (long long) live_blocks.size(),
                                (long long) cached_bytes[device].live);

                    cached_blocks.erase(block_itr);

                    block_itr++;
                }

                // Unlock
                mutex.Unlock();

                // Return under error
                if (error) return error;

                // Try to allocate again
                if (CubDebug(error = cudaMallocHost(&search_key.d_ptr, search_key.bytes))) return error;
            }

            // Create ready event
            if (CubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
                return error;

            // Insert into live blocks
            mutex.Lock();
            live_blocks.insert(search_key);
            cached_bytes[device].live += search_key.bytes;
            mutex.Unlock();

            if (debug)
                _CubLog("\tDevice %d allocated new device block at %p (%lld bytes associated with stream %lld).\n",
                        device, search_key.d_ptr, (long long) search_key.bytes,
                        (long long) search_key.associated_stream);

            // Attempt to revert back to previous device if necessary
            if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device)) {
                if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
            }
        }

        // Copy device pointer to output parameter
        *d_ptr = search_key.d_ptr;

        if (debug)
            _CubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
                    (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
                    (long long) live_blocks.size(), (long long) cached_bytes[device].live);

        return error;
    }

    cudaError_t HostAllocator::DeviceAllocate(void **d_ptr, size_t bytes, cudaStream_t active_stream) {
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    }

    cudaError_t HostAllocator::DeviceFree(int device, void *d_ptr) {
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
            if (CubDebug(error = cudaGetDevice(&entrypoint_device)))
                return error;
            device = entrypoint_device;
        }

        // Lock
        mutex.Lock();

        // Find corresponding block descriptor
        bool recached = false;
        BlockDescriptor search_key(d_ptr, device);
        BusyBlocks::iterator block_itr = live_blocks.find(search_key);
        if (block_itr != live_blocks.end()) {
            // Remove from live blocks
            search_key = *block_itr;
            live_blocks.erase(block_itr);
            cached_bytes[device].live -= search_key.bytes;

            // Keep the returned allocation if bin is valid and we won't exceed the max cached threshold
            if ((search_key.bin != INVALID_BIN) && (cached_bytes[device].free + search_key.bytes <= max_cached_bytes)) {
                // Insert returned allocation into free blocks
                recached = true;
                cached_blocks.insert(search_key);
                cached_bytes[device].free += search_key.bytes;

                if (debug)
                    _CubLog("\tDevice %d returned %lld bytes from associated stream %lld.\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
                            device, (long long) search_key.bytes, (long long) search_key.associated_stream,
                            (long long) cached_blocks.size(),
                            (long long) cached_bytes[device].free, (long long) live_blocks.size(),
                            (long long) cached_bytes[device].live);
            }
        }

        // Unlock
        mutex.Unlock();

        // First set to specified device (entrypoint may not be set)
        if (device != entrypoint_device) {
            if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
            if (CubDebug(error = cudaSetDevice(device))) return error;
        }

        if (recached) {
            // Insert the ready event in the associated stream (must have current device set properly)
            if (CubDebug(error = cudaEventRecord(search_key.ready_event, search_key.associated_stream))) return error;
        } else {
            // Free the allocation from the runtime and cleanup the event.
            if (CubDebug(error = cudaFreeHost(d_ptr))) return error;
            if (CubDebug(error = cudaEventDestroy(search_key.ready_event))) return error;

            if (debug)
                _CubLog("\tDevice %d freed %lld bytes from associated stream %lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                        device, (long long) search_key.bytes, (long long) search_key.associated_stream,
                        (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
                        (long long) live_blocks.size(), (long long) cached_bytes[device].live);
        }

        // Reset device
        if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device)) {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;
    }

    cudaError_t HostAllocator::DeviceFree(void *d_ptr) {
        return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
    }

    cudaError_t HostAllocator::FreeAllCached() {
        cudaError_t error = cudaSuccess;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        int current_device = INVALID_DEVICE_ORDINAL;

        mutex.Lock();

        while (!cached_blocks.empty()) {
            // Get first block
            CachedBlocks::iterator begin = cached_blocks.begin();

            // Get entry-point device ordinal if necessary
            if (entrypoint_device == INVALID_DEVICE_ORDINAL) {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
            }

            // Set current device ordinal if necessary
            if (begin->device != current_device) {
                if (CubDebug(error = cudaSetDevice(begin->device))) break;
                current_device = begin->device;
            }

            // Free device memory
            if (CubDebug(error = cudaFreeHost(begin->d_ptr))) break;
            if (CubDebug(error = cudaEventDestroy(begin->ready_event))) break;

            // Reduce balance and erase entry
            cached_bytes[current_device].free -= begin->bytes;

            if (debug)
                _CubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                        current_device, (long long) begin->bytes, (long long) cached_blocks.size(),
                        (long long) cached_bytes[current_device].free, (long long) live_blocks.size(),
                        (long long) cached_bytes[current_device].live);

            cached_blocks.erase(begin);
        }

        mutex.Unlock();

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL) {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;
    }

    HostAllocator::~HostAllocator() {
        if (!skip_cleanup) {
            std::atexit([]() {
                SyncMem::clear_cache();
            });
        }
    }
}
