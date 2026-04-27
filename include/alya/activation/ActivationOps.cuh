#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <type_traits>

#include <alya/core/precision/PrecisionUtils.cuh>

template <typename T>
__host__ __device__
T expVali(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return expf(x);
    } else {
        return exp(x);
    }
}

template <>
__device__ __forceinline__ 
__half expVali<__half>(__half x) { return __float2half(expf(__half2float(x))); }

template <>
__device__ __forceinline__ 
__nv_bfloat16 expVali<__nv_bfloat16>(__nv_bfloat16 x) { return __float2bfloat16(expf(__bfloat162float(x))); }

template <typename T>
__host__ __device__
T tanhVali(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return tanhf(x);
    } else {
        return tanh(x);
    }
}

template <>
__device__ __forceinline__ 
__half tanhVali<__half>(__half x) { return __float2half(tanf(__half2float(x))); }

template <>
__device__ __forceinline__
__nv_bfloat16 tanhVali<__nv_bfloat16>(__nv_bfloat16 x) { return __float2bfloat16(tanf(__bfloat162float(x))); }

/// @brief Linear activation
/// @note f(x) = x
/// @note f'(x) = 1
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct LinearOp {
    __host__ __device__ static T apply(T x) { return x; }

    __host__  __device__ static T derivativeFromOutput(T y) { return gpuOne<T>(); }
};

/// @brief ReLu activation
/// @note f(x) = (x > 0) ? x : 0
/// @note f'(x) = (y > 0) ? 1 : 0
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct ReLuOp {
    __host__ __device__ 
    static T apply(T x) { return gpuGt(x, gpuZero<T>()) ? x : gpuZero<T>(); }

    __host__ __device__ 
    static T derivativeFromOutput(T y) { return gpuGt(y, gpuZero<T>()) ? gpuOne<T>() : gpuZero<T>(); }
};

/// @brief LeakyReLu activation | has alpha value
/// @note f(x) = (x > 0) ? x : alpha*x
/// @note f'(x) = (y > 0) ? 1 : alpha
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct LeakyReLuOp {
    __host__ __device__ 
    static T apply(T x) {
        T alpha = gpuLeakyAlpha<T>();
        return gpuGt(x, gpuZero<T>()) ? x : gpuMul(alpha, x);
    }
    
    __host__ __device__ 
    static T derivativeFromOutput(T y) {
        return gpuGt(y, gpuZero<T>()) ? gpuOne<T>() : gpuLeakyAlpha<T>();
    }
};

/// @brief ELU activation
/// @note f(x) = (x > 0) ? x : e^x - 1
/// @note f'(x) = (y > 0) ? 1 : y - 0
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct ELUOp {
    __host__ __device__ 
    static T apply(T x) {
        return gpuGt(x, gpuZero<T>()) ? x : gpuSub(expVali(x), gpuOne<T>());
    }

    __host__ __device__ 
    static T derivativeFromOutput(T y) {
        return gpuGt(y, gpuZero<T>()) ? gpuOne<T>() : gpuSub(y, gpuZero<T>());
    }
};

/// @brief Sigmoid activation | has alpha value
/// @note f(x) = (1 + e^(-x)) / 1
/// @note f'(x) = (1 - y) * y
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct SigmoidOp {
    __host__ __device__ 
    static T apply(T x) {
        return gpuDiv(gpuOne<T>(), gpuAdd(gpuOne<T>(), expVali(gpuNeg(x))));
    }

    __host__ __device__ 
    static T derivativeFromOutput(T y) {
        return gpuMul(y, gpuSub(gpuOne<T>(), y));
    }
};

/// @brief Tanh activation
/// @note f(x) = tanh(x)
/// @note f'(x) = 1 - y^2
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct TanhOp {
    __host__ __device__ 
    static T apply(T x) { return tanhVali(x); }

    __host__ __device__ 
    static T derivativeFromOutput(T y) {
        return gpuSub(gpuOne<T>(), gpuMul(y, y));
    }
};