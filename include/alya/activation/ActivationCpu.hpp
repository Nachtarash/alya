#pragma once

#include <cstddef>

#include <alya/activation/ActivationOps.cuh> 

namespace activationCpu {

    template <typename Op, typename T>
    void applyCpu(const T* in, T* out, size_t N) {
        for(size_t i = 0; i < N; i++) {
            out[i] = Op::apply(in[i]);
        }
    }

    template <typename Op, typename T>
    void derivativeCpu(const T* y, T* out, size_t N) {
        for(size_t i = 0; i < N; i++) {
            out[i] = Op::derivativeFromOutput(y[i]);
        }
    }
}
