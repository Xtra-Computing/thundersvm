//
// Created by jiashuai on 17-9-14.
//

#ifndef THUNDERSVM_CLION_CUDA_H
#define THUNDERSVM_CLION_CUDA_H

#include <thundersvm/config.h>
#ifdef __JETBRAINS_IDE__
#ifdef USE_CUDA
#include "math.h"
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__
// CUDA Synchronization
inline void __syncthreads() {};
inline void __threadfence_block() {};
inline void __threadfence() {};
inline void __threadfence_system();
inline int __syncthreads_count(int predicate) {return predicate};
inline int __syncthreads_and(int predicate) {return predicate};
inline int __syncthreads_or(int predicate) {return predicate};
template<class T> inline T __clz(const T val) { return val; }
template<class T> inline T __ldg(const T* address){return *address};
// CUDA TYPES
typedef unsigned short uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef long long longlong;

typedef struct uchar1{
    uchar x;
}uchar1;

typedef struct uchar2{
    uchar x;
    uchar y;
}uchar2;

typedef struct uchar3{
    uchar x;
    uchar y;
    uchar z;
}uchar3;

typedef struct uchar4{
    uchar x;
    uchar y;
    uchar z;
    uchar w;
}uchar4;

typedef struct char1{
    char x;
}char1;

typedef struct char2{
    char x;
    char y;
}char2;

typedef struct char3{
    char x;
    char y;
    char z;
}char3;

typedef struct char4{
    char x;
    char y;
    char z;
    char w;
}char4;

typedef struct ushort1{
    ushort x;
}ushort1;

typedef struct ushort2{
    ushort x;
    ushort y;
}ushort2;

typedef struct ushort3{
    ushort x;
    ushort y;
    ushort z;
}ushort3;

typedef struct ushort4{
    ushort x;
    ushort y;
    ushort z;
    ushort w;
}ushort4;

typedef struct short1{
    short x;
}short1;

typedef struct short2{
    short x;
    short y;
}short2;

typedef struct short3{
    short x;
    short y;
    short z;
}short3;

typedef struct short4{
    short x;
    short y;
    short z;
    short w;
}short4;

typedef struct uint1{
    uint x;
}uint1;

typedef struct uint2{
    uint x;
    uint y;
}uint2;

typedef struct uint3{
    uint x;
    uint y;
    uint z;
}uint3;

typedef struct uint4{
    uint x;
    uint y;
    uint z;
    uint w;
}uint4;

typedef struct int1{
    int x;
}int1;

typedef struct int2{
    int x;
    int y;
}int2;

typedef struct int3{
    int x;
    int y;
    int z;
}int3;

typedef struct int4{
    int x;
    int y;
    int z;
    int w;
}int4;

typedef struct ulong1{
    ulong x;
}ulong1;

typedef struct ulong2{
    ulong x;
    ulong y;
}ulong2;

typedef struct ulong3{
    ulong x;
    ulong y;
    ulong z;
}ulong3;

typedef struct ulong4{
    ulong x;
    ulong y;
    ulong z;
    ulong w;
}ulong4;

typedef struct long1{
    long x;
}long1;

typedef struct long2{
    long x;
    long y;
}long2;

typedef struct long3{
    long x;
    long y;
    long z;
}long3;

typedef struct long4{
    long x;
    long y;
    long z;
    long w;
}long4;

typedef struct ulonglong1{
    ulonglong x;
}ulonglong1;

typedef struct ulonglong2{
    ulonglong x;
    ulonglong y;
}ulonglong2;

typedef struct ulonglong3{
    ulonglong x;
    ulonglong y;
    ulonglong z;
}ulonglong3;

typedef struct ulonglong4{
    ulonglong x;
    ulonglong y;
    ulonglong z;
    ulonglong w;
}ulonglong4;

typedef struct longlong1{
    longlong x;
}longlong1;

typedef struct longlong2{
    longlong x;
    longlong y;
}longlong2;

typedef struct float1{
    float x;
}float1;

typedef struct float2{
    float x;
    float y;
}float2;

typedef struct float3{
    float x;
    float y;
    float z;
}float3;

typedef struct float4{
    float x;
    float y;
    float z;
    float w;
}float4;

typedef struct double1{
    double x;
}double1;

typedef struct double2{
    double x;
    double y;
}double2;

typedef uint3 dim3;

extern dim3 gridDim;
extern uint3 blockIdx;
extern dim3 blockDim;
extern uint3 threadIdx;
extern int warpsize;
#endif
#endif
#endif //THUNDERSVM_CLION_CUDA_H
