#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>

inline double toCompute(double x) {
    return x;
}

inline float toCompute(float x) {
    return x;
}

inline float toCompute(__half x) {
    return __half2float(x);
}

inline float toCompute(__nv_bfloat16 x) {
    return __bfloat162float(x);
}


__host__ __device__ 
inline double toStorage(double x) {
    return x;
}

template <typename T>
__host__ __device__ 
T toStorage(float x);

template <>
__host__ __device__ 
inline float toStorage<float>(float x) {
    return x;
}

template <>
__host__ __device__ 
inline double toStorage<double>(float x) {
    return static_cast<double>(x);
}

template <>
__host__ __device__ 
inline __half toStorage<__half>(float x) {
    return __float2half(x);
}

template <>
__host__ __device__ 
inline __nv_bfloat16 toStorage<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}


template <typename T> 
__host__ __device__ __forceinline__ 
T gpuZero();

template <>
__host__ __device__ __forceinline__ 
float gpuZero<float>() { return 0.0f; }

template <>
__host__ __device__ __forceinline__ 
double gpuZero<double>() { return 0.0; }

template <>
__device__ __forceinline__ 
__half gpuZero<__half>() { return __float2half(0.0f); }

template <>
__device__ __forceinline__ 
__nv_bfloat16 gpuZero<__nv_bfloat16>() { return __float2bfloat16(0.0f); }



template <typename T>
__host__ __device__ __forceinline__ 
T gpuOne();

template <>
__host__ __device__ __forceinline__ 
float gpuOne<float>() { return 1.0f; }

template <>
__host__ __device__ __forceinline__ 
double gpuOne<double>() { return 1.0; }

template <>
__device__ __forceinline__ 
__half gpuOne<__half>() { return __float2half(1.0f); }

template <>
__device__ __forceinline__ 
__nv_bfloat16 gpuOne<__nv_bfloat16>() { return __float2bfloat16(1.0f); }



template <typename T>
__host__ __device__ __forceinline__ 
T gpuLeakyAlpha();

template <> 
__host__ __device__ __forceinline__ 
float gpuLeakyAlpha<float>() { return 0.001f; }

template <> 
__host__ __device__ __forceinline__ 
double gpuLeakyAlpha<double>() { return 0.001; }

template <> 
__device__ __forceinline__ 
__half gpuLeakyAlpha<__half>() { return __float2half(0.001f); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuLeakyAlpha<__nv_bfloat16>() { return __float2bfloat16(0.001f); }



template <typename T> 
__host__ __device__ __forceinline__ 
bool gpuGt(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
bool gpuGt<float>(float a, float b) { return a > b; }

template <> 
__host__ __device__ __forceinline__ 
bool gpuGt<double>(double a, double b){ return a > b; }

template <> 
__device__ __forceinline__ 
bool gpuGt<__half>(__half a, __half b) { return __hgt(a, b); }

template <> 
__device__ __forceinline__ 
bool gpuGt<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hgt(a, b); }



template <typename T> 
__host__ __device__ __forceinline__ 
bool gpuLt(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
bool gpuLt<float>(float a, float b) { return a < b; }

template <> 
__host__ __device__ __forceinline__ 
bool gpuLt<double>(double a, double b){ return a < b; }

template <> 
__device__ __forceinline__ 
bool gpuLt<__half>(__half a, __half b) { return __hlt(a, b); }

template <> 
__device__ __forceinline__ 
bool gpuLt<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hlt(a, b); }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuMul(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
float gpuMul<float>(float a, float b) { return a * b; }

template <> 
__host__ __device__ __forceinline__ 
double gpuMul<double>(double a, double b) { return a * b; }

template <> 
__device__ __forceinline__ 
__half gpuMul<__half>(__half a, __half b) { return __hmul(a, b); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuMul<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuDiv(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
float gpuDiv<float>(float a, float b) { return a / b; }

template <> 
__host__ __device__ __forceinline__ 
double gpuDiv<double>(double a, double b) { return a / b; }

template <> 
__device__ __forceinline__ 
__half gpuDiv<__half>(__half a, __half b) { return __hdiv(a, b); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuDiv<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hdiv(a, b); }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuSub(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
float gpuSub<float>(float a, float b) { return a - b; }

template <> 
__host__ __device__ __forceinline__ 
double gpuSub<double>(double a, double b) { return a - b; }

template <> 
__device__ __forceinline__ 
__half gpuSub<__half>(__half a, __half b) { return __hsub(a, b); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuSub<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hsub(a, b); }

template <typename T> 
__host__ __device__ __forceinline__ 
T gpuAdd(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
float gpuAdd<float>(float a, float b) { return a + b; }

template <> 
__host__ __device__ __forceinline__ 
double gpuAdd<double>(double a, double b) { return a + b; }

template <> 
__device__ __forceinline__ 
__half gpuAdd<__half>(__half a, __half b) { return __hadd(a, b); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuAdd<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuNeg(T x);

template <> 
__host__ __device__ __forceinline__ 
float gpuNeg<float>(float x) { return -x; }

template <> 
__host__ __device__ __forceinline__ 
double gpuNeg<double>(double x) { return -x; }

template <> 
__device__ __forceinline__ 
__half gpuNeg<__half>(__half x) { return __hneg(x); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuNeg<__nv_bfloat16>(__nv_bfloat16 x) { return __hneg(x); }


template <typename T>
__host__ __device__ __forceinline__ 
T gpuMin(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
float gpuMin<float>(float a, float b) { return a < b ? a : b; }

template <> 
__host__ __device__ __forceinline__ 
double gpuMin<double>(double a, double b) { return a < b ? a : b; }

template <> 
__device__ __forceinline__ 
__half gpuMin<__half>(__half a, __half b) { return __hlt(a, b) ? a : b; }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuMin<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hlt(a, b) ? a : b; }


template <typename T>
__host__ __device__ __forceinline__ 
T gpuMax(T a, T b);

template <> 
__host__ __device__ __forceinline__ 
float gpuMax<float>(float a, float b) { return a > b ? a : b; }

template <> 
__host__ __device__ __forceinline__ 
double gpuMax<double>(double a, double b) { return a > b ? a : b; }

template <> 
__device__ __forceinline__ 
__half gpuMax<__half>(__half a, __half b) { return __hgt(a, b) ? a : b; }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuMax<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hgt(a, b) ? a : b; }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuSqrt(T x);

template <> 
__host__ __device__ __forceinline__ 
float gpuSqrt<float>(float x) { return sqrtf(x); }

template <> 
__host__ __device__ __forceinline__ 
double gpuSqrt<double>(double x) { return sqrt(x); }

template <> 
__device__ __forceinline__ 
__half gpuSqrt<__half>(__half x) { return __float2half(sqrtf(__half2float(x))); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuSqrt<__nv_bfloat16>(__nv_bfloat16 x) { return __float2bfloat16(sqrtf(__bfloat162float(x))); }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuAbs(T x);

template <> 
__host__ __device__ __forceinline__ 
float gpuAbs<float>(float x) { return fabsf(x); }

template <> 
__host__ __device__ __forceinline__ 
double gpuAbs<double>(double x) { return fabs(x); }

template <> 
__device__ __forceinline__ 
__half gpuAbs<__half>(__half x) { return __habs(x); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuAbs<__nv_bfloat16>(__nv_bfloat16 x) { return __habs(x); }



template <typename T> 
__host__ __device__ __forceinline__ 
T gpuLog(T x);

template <> 
__host__ __device__ __forceinline__ 
float gpuLog<float>(float x) { return logf(x); }

template <> 
__host__ __device__ __forceinline__ 
double gpuLog<double>(double x) { return log(x); }

template <> 
__device__ __forceinline__ 
__half gpuLog<__half>(__half x) { return __float2half(logf(__half2float(x))); }

template <> 
__device__ __forceinline__ 
__nv_bfloat16 gpuLog<__nv_bfloat16>(__nv_bfloat16 x) { return __float2bfloat16(logf(__bfloat162float(x))); }
