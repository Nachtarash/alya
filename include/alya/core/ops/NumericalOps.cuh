#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
struct MulOp {
    __host__ __device__ __forceinline__
    static T apply(T a, T b) { return a * b; }
};

template <>
struct MulOp<__half> {
    __device__ __forceinline__
    static __half apply(__half a, __half b) { return __hmul(a, b); }
};

template <>
struct MulOp<__nv_bfloat16> {
    __device__ __forceinline__
    static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
};

template <typename T>
struct DivOp {
    __host__ __device__ __forceinline__
    static T apply(T a, T b) { return a / (b + T(1e-6)); }
};


template <>
struct DivOp<__half> {
    __device__ __forceinline__
    static __half apply(__half a, __half b) { 
        __half eps = __float2half(1e-6);

        return __hdiv(a, __hadd(b, eps)); }
};

template <>
struct DivOp<__nv_bfloat16> {
    __device__ __forceinline__
    static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b) {
        __nv_bfloat16 eps = __float2bfloat16(1e-6);

        return __hdiv(a, __hadd(b, eps)); }
};


template <typename T>
struct AddOp {
    __host__ __device__ __forceinline__
    static T apply(T a, T b) { return a + b; }
};

template <>
struct AddOp<__half> {
    __device__ __forceinline__
    static __half apply(__half a, __half b) { return __hadd(a, b); }
};

template <>
struct AddOp<__nv_bfloat16> {
    __device__ __forceinline__
    static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
};


template <typename T>
struct SubOp {
    __host__ __device__ __forceinline__
    static T apply(T a, T b) { return a - b; }
};

template <>
struct SubOp<__half> {
    __device__ __forceinline__
    static __half apply(__half a, __half b) { return __hsub(a, b); }
};

template <>
struct SubOp<__nv_bfloat16> {
    __device__ __forceinline__
    static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }
};


template <typename T>
struct SquareOp {
    __host__ __device__ __forceinline__
    static T apply(T a) { return a * a; }
};

template <>
struct SquareOp<__half> {
    __device__ __forceinline__
    static __half apply(__half a) { return __hmul(a, a); }
};

template <>
struct SquareOp<__nv_bfloat16> {
    __device__ __forceinline__
    static __nv_bfloat16 apply(__nv_bfloat16 a) { return __hmul(a, a); }
};


template <typename T>
struct SqrtOp {
    __host__ __device__ __forceinline__
    static T apply(T a) { return sqrt(a); }
};

template <>
struct SqrtOp<__half> {
    __device__ __forceinline__
    static __half apply(__half a) { return __float2half(sqrtf(__half2float(a))); }
};

template <>
struct SqrtOp<__nv_bfloat16> {
    __device__ __forceinline__
    static __nv_bfloat16 apply(__nv_bfloat16 a) { return __float2bfloat16(sqrtf(__bfloat162float(a))); }
};