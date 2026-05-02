#pragma once
 
#include <cstddef>

#include <alya/activation/ActivationOps.cuh> 

namespace activationCpu {

    template <typename Op, typename T>
    void applyCpu(const T* in, T* out, const size_t N) {
        for(size_t i = 0; i < N; i++) {
            out[i] = Op::apply(in[i]);
        }
    }

    template <typename Op, typename T>
    void backwardCpu(const T* gradOut, const T* z, const T* a, T* gradZ, const size_t N) {
        for(size_t i = 0; i < N; i++) {
            gradZ[i] = Op::backwardScalar(gradOut[i], z[i], a[i]);
        }
    }
}
