#pragma once

#include <cstdint>

namespace alya {

/// @brief Options for Device 
enum class DeviceType : uint8_t {
    CPU = 0,
    GPU = 1
};

/// @brief Device on which Model should run
struct Device {
    DeviceType type = DeviceType::CPU;
    uint8_t gpu_id = 0;
};

}   //namespace alya