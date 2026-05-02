#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <type_traits>
#include <cmath>

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


template <typename T>
struct ComputeType {
    using type = T;
};

template <>
struct ComputeType<__half> {
    using type = float;
};

template <>
struct ComputeType<__nv_bfloat16> {
    using type = float;
};

template <typename T>
using compute_t = typename ComputeType<T>::type;

namespace activationDetail {

template <typename T>
__host__ __device__
compute_t<T> storageToCompute(T x);

template <>
__host__ __device__
inline float storageToCompute<float>(float x) { return x; }

template <>
__host__ __device__
inline double storageToCompute<double>(double x) { return x; }

template <>
__device__ __forceinline__
float storageToCompute<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__
float storageToCompute<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T, typename ComputeT>
__host__ __device__
T fromCompute(ComputeT x);

template <>
__host__ __device__
inline float fromCompute<float, float>(float x) { return x; }

template <>
__host__ __device__
inline double fromCompute<double, double>(double x) { return x; }

template <>
__host__ __device__
inline double fromCompute<double, float>(float x) { return static_cast<double>(x); }

template <>
__host__ __device__
inline float fromCompute<float, double>(double x) { return static_cast<float>(x); }

template <>
__host__ __device__
inline __half fromCompute<__half, float>(float x) { return toStorage<__half>(x); }

template <>
__host__ __device__
inline __nv_bfloat16 fromCompute<__nv_bfloat16, float>(float x) { return toStorage<__nv_bfloat16>(x); }

template <typename T>
__host__ __device__
compute_t<T> geluApplyCompute(compute_t<T> x) {
    const T k = static_cast<T>(0.7978845608); // sqrt(2/pi)
    const T c = static_cast<T>(0.044715);

    T x3 = x * x * x;
    T inner = k * (x + c * x3);
    T t = tanhVali(inner);

    return static_cast<T>(0.5) * x * (static_cast<T>(1) + t);
}

template <typename T>
__host__ __device__
compute_t<T> geluDerivativeCompute(compute_t<T> x) {
    const T k = static_cast<T>(0.7978845608); // sqrt(2/pi)
    const T c = static_cast<T>(0.044715);

    T x2 = x * x;
    T x3 = x2 * x;
    T inner = k * (x + c * x3);
    T t = tanhVali(inner);

    T term1 = static_cast<T>(0.5) * (static_cast<T>(1) + t);
    T partial1 = static_cast<T>(1) - t * t;
    T partial2 = static_cast<T>(1) + (static_cast<T>(3) * c * x2);
    T term2 = static_cast<T>(0.5) * x * partial1 * k * partial2;

    return term1 + term2;
}

//for swish
template <typename T>
__host__ __device__
compute_t<T> sigmoidCompute(compute_t<T> x) {
    return static_cast<T>(1) / (static_cast<T>(1) + expVali(-x));
}

template <typename ComputeT>
struct GELUFormula {
    __host__ __device__
    static ComputeT apply(ComputeT x) {
        return geluApplyCompute<ComputeT>(x);
    }

    __host__ __device__
    static ComputeT derivative(ComputeT x) {
        return geluDerivativeCompute<ComputeT>(x);
    }
};

template <typename ComputeT>
struct SwishFormula {
    __host__ __device__
    static ComputeT apply(ComputeT x) {
        return x * sigmoidCompute<ComputeT>(x);
    }

    __host__ __device__
    static ComputeT derivative(ComputeT x) {
        ComputeT s = sigmoidCompute<ComputeT>(x);
        return s + x * s * (static_cast<ComputeT>(1) - s);
    }
};

template <typename T, template <typename> class Formula>
__host__ __device__
T applyWithCompute(T x) {
    using ComputeT = compute_t<T>;
    ComputeT xc = storageToCompute(x);
    ComputeT yc = Formula<ComputeT>::apply(xc);
    return fromCompute<T>(yc);
}

template <typename T, template <typename> class Formula>
__host__ __device__
T backwardWithCompute(T gradOut, T x) {
    using ComputeT = compute_t<T>;
    ComputeT xc = storageToCompute(x);
    ComputeT gc = storageToCompute(gradOut);
    ComputeT out = gc * Formula<ComputeT>::derivative(xc);
    return fromCompute<T>(out);
}

}   //namespace activationDetail
