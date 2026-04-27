#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include <alya/activation/ActivationOps.cuh>

namespace activationGpu {
    
    template <typename Op, typename T>
    void applyGpu(const T* in, T* out, size_t N);

    template <typename Op, typename T>
    void derivativeGpu(const T* y, T* out, size_t N);

}