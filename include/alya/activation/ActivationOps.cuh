#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/activation/ActivationUtils.cuh>

/// @brief Linear activation
/// @note f(x) = x
/// @note f'(x) = 1
/// @note derivative uses output y from f(x) for f'(x)
template <typename T>
struct LinearOp {
    __host__ __device__ static T apply(T x) { return x; }

    __host__  __device__ static T backwardScalar(T gradOut, T z, T a) {
        return gradOut;
    }
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
    static T backwardScalar(T gradOut, T z, T a) {
        return gpuGt(a, gpuZero<T>()) ? gradOut : gpuZero<T>();
    }
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
    static T backwardScalar(T gradOut, T z, T a) {
        T slope = gpuGt(a, gpuZero<T>()) ? gpuOne<T>() : gpuLeakyAlpha<T>();
        return gpuMul(gradOut, slope);
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
    static T backwardScalar(T gradOut, T z, T a) {
        T slope = gpuGt(a, gpuZero<T>()) ? gpuOne<T>() : gpuAdd(a, gpuOne<T>());
        return gpuMul(gradOut, slope);
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
    static T backwardScalar(T gradOut, T z, T a) {
        return gpuMul(gradOut, gpuMul(a, gpuSub(gpuOne<T>(), a)));
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
    static T backwardScalar(T gradOut, T z, T a) {
        return gpuMul(gradOut, gpuSub(gpuOne<T>(), gpuMul(a, a)));
    }
};

/// @brief GLEU activation (approximate)
/// @note f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// @note f'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
/// @note u = sqrt(2/π) * (x + 0.044715 * x^3)  
template <typename T>
struct GELUOp {
    __host__ __device__
    static T apply(T x) {
        return activationDetail::applyWithCompute<T, activationDetail::GELUFormula>(x);
    }

    __host__ __device__
    static T backwardScalar(T gradOut, T x, T a) {
        return activationDetail::backwardWithCompute<T, activationDetail::GELUFormula>(gradOut, x);
    }
};

/// @brief Swish activation
/// @note f(x) = x * sigmoid(x)
/// @note f'(x) = f(x) + sigmoid(x) * (1 - f(x))
template <typename T>
struct SwishOp {
    __host__ __device__
    static T apply(T x) {
        return activationDetail::applyWithCompute<T, activationDetail::SwishFormula>(x);
    }

    __host__ __device__
    static T backwardScalar(T gradOut, T x, T a) {
        return activationDetail::backwardWithCompute<T, activationDetail::SwishFormula>(gradOut, x);
    }
};
