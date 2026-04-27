#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__device__ __forceinline__ 
T gpuMaxVal();

template <>
__device__ __forceinline__ 
float gpuMaxVal<float>() { return 3.402823466e+38f; }

template <>
__device__ __forceinline__ 
double gpuMaxVal<double>() { return 1.7976931348623157e+308; }

template <>
__device__ __forceinline__ 
__half gpuMaxVal<__half>() { return __float2half(65504.0f); }

template <>
__device__ __forceinline__ 
__nv_bfloat16 gpuMaxVal<__nv_bfloat16>() { return __float2bfloat16(3.38953139e+38f); }

template <typename T>
__device__ __forceinline__ 
T gpuMinVal();

template <>
__device__ __forceinline__ 
float gpuMinVal<float>() { return -3.402823466e+38f; }

template <> 
__device__ __forceinline__ 
double gpuMinVal<double>() { return -1.7976931348623157e+308; }

template <>
__device__ __forceinline__ 
__half gpuMinVal<__half>() { return __float2half(-65504.0f); }

template <>
__device__ __forceinline__ 
__nv_bfloat16 gpuMinVal<__nv_bfloat16>() { return __float2bfloat16(-3.38953139e+38f); }