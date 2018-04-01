//
// Created by jiashuai on 17-9-16.
//
#include "gtest/gtest.h"
#include "thundersvm/syncmem.h"

TEST(SyncMemTest, host_allocate){
//    EXPECT_EQ(SyncMem::total_memory_size, 0);
    //one instance
    SyncMem syncMem(100);
    EXPECT_NE(syncMem.host_data(), nullptr);
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::HOST);
    EXPECT_EQ(syncMem.size(), 100);
    EXPECT_EQ(SyncMem::total_memory_size, 100);

    //one instance without initialization
    SyncMem syncMem2(100);
    EXPECT_EQ(SyncMem::total_memory_size, 100);
    syncMem2.to_host();

    //one instance with initialization
    EXPECT_EQ(SyncMem::total_memory_size, 200);

    //set data doesn't increase total size
    {
        SyncMem syncMem_t(100);
        syncMem_t.set_host_data(syncMem2.host_data());
        EXPECT_EQ(SyncMem::total_memory_size, 200);
    }
    EXPECT_EQ(SyncMem::total_memory_size, 200);

    //set data doesn't increase total size
    {
        SyncMem syncMem_t(100);
        syncMem_t.to_host();
        EXPECT_EQ(SyncMem::total_memory_size, 300);
        syncMem_t.set_host_data(syncMem2.host_data());
        EXPECT_EQ(SyncMem::total_memory_size, 200);
    }
    EXPECT_EQ(SyncMem::total_memory_size, 200);

    //deconstruction
    {
        SyncMem syncMem_t(100);
        syncMem_t.to_host();
        EXPECT_EQ(SyncMem::total_memory_size, 300);
    }
    EXPECT_EQ(SyncMem::total_memory_size, 200);
}
#ifdef USE_CUDA
TEST(SyncMemTest, device_allocate){
    SyncMem syncMem(100);
    EXPECT_NE(syncMem.device_data(), nullptr);
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::DEVICE);
    EXPECT_EQ(syncMem.size(), 100);
    EXPECT_EQ(syncMem.get_total_memory_size(), 100);

    //one instance without initialization
    SyncMem syncMem2(100);
    EXPECT_EQ(SyncMem::get_total_memory_size(), 100);
    syncMem2.to_host();
    syncMem2.to_device();

    //one instance with initialization
    EXPECT_EQ(SyncMem::get_total_memory_size(), 200);

    //set data doesn't increase total size
    {
        SyncMem syncMem_t(100);
        syncMem_t.set_host_data(syncMem2.host_data());
        syncMem_t.set_device_data(syncMem2.device_data());
        EXPECT_EQ(SyncMem::get_total_memory_size(), 200);
    }
    EXPECT_EQ(SyncMem::get_total_memory_size(), 200);

    //set data doesn't increase total size
    {
        SyncMem syncMem_t(100);
        syncMem_t.to_host();
        syncMem.to_device();
        EXPECT_EQ(SyncMem::get_total_memory_size(), 300);
        syncMem_t.set_host_data(syncMem2.host_data());
        syncMem_t.set_device_data(syncMem2.device_data());
        EXPECT_EQ(SyncMem::get_total_memory_size(), 200);
    }
    EXPECT_EQ(SyncMem::get_total_memory_size(), 200);

    //deconstruction
    {
        SyncMem syncMem_t(100);
        syncMem_t.to_host();
        syncMem_t.to_device();
        EXPECT_EQ(SyncMem::get_total_memory_size(), 300);
    }
    EXPECT_EQ(SyncMem::get_total_memory_size(), 200);
}

TEST(SyncMemTest, host_to_device){
    SyncMem syncMem(sizeof(int) * 10);
    int *data = static_cast<int *>(syncMem.host_data());
    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }
    syncMem.to_device();
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::DEVICE);
    for (int i = 0; i < 10; ++i) {
        data[i] = -1;
    }
    syncMem.to_host();
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::HOST);
    data = static_cast<int *>(syncMem.host_data());
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i] , i);
    }
}

#endif

