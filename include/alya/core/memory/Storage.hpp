#pragma once

#include <cstddef>
#include <cstdint>

#include <alya/core/memory/Device.hpp>

namespace alya {

struct Storage {
    void* cpuPtr = nullptr;
    void* gpuPtr = nullptr;

    bool cpuValid = false;
    bool gpuValid = false;

    size_t bytes = 0;
    uint16_t refcount;
    
    Device device;
};

}   //namespace alya
