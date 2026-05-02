#pragma once

#include <cstdlib>

#include <alya/profiling/print.hpp>

#ifdef NDEBUG
    #define CUDA_CHECK(call) (call)
#else
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = (call); \
            if (err != cudaSuccess) { \
                alya::eprint("CUDA error in ", __func__, \
                             " (", __FILE__, ":", __LINE__, "): ", \
                             cudaGetErrorString(err), alya::Endl{}); \
                std::exit(1); \
            } \
        } while(0)
#endif
