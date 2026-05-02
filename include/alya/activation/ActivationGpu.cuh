#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include <alya/activation/ActivationOps.cuh>

namespace activationGpu {
    
    template <typename Op, typename T>
    void applyGpu(const T* in, T* out, size_t N);

    template <typename Op, typename T>
    void backwardGpu(const T* gradOut, const T* z, const T* a, T* gradZ, size_t N);

}
