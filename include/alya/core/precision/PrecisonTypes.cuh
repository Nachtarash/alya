#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>

namespace alya {

/// @brief fp64 precision type
/// @note storage: double (64bit)
/// @note compute: double (64bit)
struct fp64 {};

/// @brief fp32 precision type
/// @note storage: float  (32bit)
/// @note compute: float  (32bit)
struct fp32 {};

/// @brief fp16 precision type
/// @note storage: __half (16bit)
/// @note compute: float (32bit)
struct fp16 {};

/// @brief bf16 precision type
/// @note storage: __nv__bfloat16 (16bit)
/// @note compute: float  (32bit)
struct bf16 {};

/// @brief index type | not for reggular computation
/// @note storage: size_t
/// @note compute: size_t
struct IndexType {};

template <typename P>
struct Precision;

template <>
struct Precision<fp64> {
    using storageT = double;
    using computeT = double;
};

template <>
struct Precision<fp32> {
    using storageT = float;
    using computeT = float;
};

template <>
struct Precision<fp16> {
    using storageT = __half;
    using computeT = float;
};

template <>
struct Precision<bf16> {
    using storageT = __nv_bfloat16;
    using computeT = float;
};

template <>
struct Precision<size_t> {
    using storageT = size_t;
    using computeT = size_t;
};

}   //namespace alya
