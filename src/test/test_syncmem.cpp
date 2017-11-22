//
// Created by jiashuai on 17-9-16.
//
#include "gtest/gtest.h"
#include "thundersvm/syncmem.h"

TEST(SyncMemTest, host_allocate){
    SyncMem syncMem(100);
    EXPECT_NE(syncMem.host_data(), nullptr);
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::HOST);
    EXPECT_EQ(syncMem.size(), 100);
}
#ifdef USE_CUDA
TEST(SyncMemTest, device_allocate){
    SyncMem syncMem(100);
    EXPECT_NE(syncMem.device_data(), nullptr);
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::DEVICE);
    EXPECT_EQ(syncMem.size(), 100);
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

