#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/core/data/ArgMinMaxContainer.hpp>

template <typename T>
__device__ __forceinline__ 
T warpReductSum(T val) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

template <typename T>
__device__ __forceinline__ 
T warpReductMin(T val) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val = gpuMin(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}

template <typename T>
__device__ __forceinline__ 
T warpReductMax(T val) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        val = gpuMax(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}

template <typename T>
__device__ __forceinline__ 
void warpReductArgmax(T& val, size_t& idx) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        T other_val = __shfl_down_sync(0xffffffff, val, offset);
        size_t other_idx = __shfl_down_sync(0xffffffff, idx, offset);

        if(other_val > val || (other_val == val && other_idx > idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <typename T>
__device__ __forceinline__ 
void warpReductArgmin(T& val, size_t& idx) {
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        T other_val = __shfl_down_sync(0xffffffff, val, offset);
        size_t other_idx = __shfl_down_sync(0xffffffff, idx, offset);

        if(other_val < val || (other_val == val && other_idx > idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}
