//
// Created by jiashuai on 17-9-16.
//
#include "gtest/gtest.h"
#include "thundersvm/syncmem.h"
TEST(SyncMemTest, host_allocate){
    SyncMem syncMem(100);
    EXPECT_NE(syncMem.cpu_data(), nullptr);
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::CPU);
}

TEST(SyncMemTest, device_allocate){
    SyncMem syncMem(100);
    EXPECT_NE(syncMem.gpu_data(), nullptr);
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::GPU);
}

TEST(SyncMemTest, host_to_device){
    SyncMem syncMem(sizeof(int) * 10);
    int *data = static_cast<int *>(syncMem.cpu_data());
    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }
    syncMem.gpu_data();
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::GPU);
    syncMem.cpu_data();
    EXPECT_EQ(syncMem.head(), SyncMem::HEAD::CPU);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i] , i);
    }
}
