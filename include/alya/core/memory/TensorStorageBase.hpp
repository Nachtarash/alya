#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <alya/core/memory/Device.hpp>
#include <alya/core/memory/Storage.hpp>
#include <alya/core/memory/TensorStorageCuda.hpp>

namespace alya::internal {

template <typename T>
class TensorStorageBase {
protected:
    mutable Storage* storage = nullptr;

    TensorStorageBase() = default;

    explicit TensorStorageBase(size_t bytes, Device dev = Device{}) {
        //std::cout << "Created tensor" << std::endl;
        storage = new Storage();
        storage -> bytes = bytes;
        storage -> device = dev;
        storage -> refcount = 1;

        if(storage -> bytes == 0) {
            return;
        }

        if(dev.type == DeviceType::GPU) {
            tensorStorageAllocateGpu(storage);
            storage -> gpuValid = false;
            storage -> cpuValid = false;
        } else {
            storage -> cpuPtr = std::malloc(storage -> bytes);

            if(!storage -> cpuPtr) {
                delete storage;
                storage = nullptr;
                throw std::runtime_error("CPU: malloc failed");
            }

            std::memset(storage -> cpuPtr, 0, storage -> bytes);
            storage -> cpuValid = true;
            storage -> gpuValid = false;
        }
    }

    ~TensorStorageBase() {
        //std::cout << "destoyed tensor" << std::endl;
        releaseStorage();
    }

    TensorStorageBase(const TensorStorageBase& other) : storage(other.storage) {
        //std::cout << "copied tensor" << std::endl;
        if(storage) {
            storage -> refcount++;
        }
    }

    TensorStorageBase(TensorStorageBase&& other) noexcept : storage(other.storage) {
        //std::cout << "moved tensor" << std::endl;
        other.storage = nullptr;
    }

    TensorStorageBase& operator=(const TensorStorageBase& other) {
        //std::cout << "assigned copied tensor" << std::endl;
        if(this == &other) {
            return *this;
        }

        releaseStorage();
        storage = other.storage;

        if(storage) {
            storage -> refcount++;
        }

        return *this;
    }

    TensorStorageBase& operator=(TensorStorageBase&& other) noexcept {
        //std::cout << "assigned moved tensor" << std::endl;
        if(this == &other) {
            return *this;
        }

        releaseStorage();
        storage = other.storage;
        other.storage = nullptr;

        return *this;
    }

    void releaseStorage() noexcept {
        if(!storage) {
            return;
        }

        if(--storage -> refcount == 0) {
            if(storage -> cpuPtr) {
                std::free(storage -> cpuPtr);
            }

            if(storage -> gpuPtr) {
                tensorStorageFreeGpu(storage);
            }

            delete storage;
        }

        storage = nullptr;
    }

public:
    void toGPU() const {
        tensorStorageSyncToGpu(storage);
    }

    void toCPU() const {
        tensorStorageSyncToCpu(storage);
    }

    T* cpuData() {
        toCPU();

        storage -> cpuValid = true;
        storage -> gpuValid = false;

        return static_cast<T*>(storage -> cpuPtr);
    }

    T* gpuData() {
        toGPU();

        storage -> gpuValid = true;
        storage -> cpuValid = false;

        return static_cast<T*>(storage -> gpuPtr);
    }

    const T* cpuData() const {
        //toCPU();

        return static_cast<T*>(storage -> cpuPtr);
    }

    const T* gpuData() const {
        //toGPU();

        return static_cast<T*>(storage -> gpuPtr);
    }

    Device device() const {
        return storage ? storage -> device : Device{};
    }

    void setDevice(const Device& dev) {
        if(storage) {
            storage -> device = dev;
        }
    }

    void printDevice() const{
        device();

        switch(storage -> device.type) {
            case DeviceType::CPU : std::cout << "Device: CPU\n" << std::endl; break;
            case DeviceType::GPU : std::cout << "Device: GPU\n" << std::endl; break;
        }
    }

    template <typename CpuFn, typename GpuFn>   //&& means universal reference (includes returnvalues etc)
    decltype(auto) deviceDispatcher(const char* fnName , CpuFn&& cpuFn, GpuFn&& gpuFn) {
        switch(storage -> device.type) {
            case DeviceType::CPU: return std::forward<CpuFn>(cpuFn)();
            case DeviceType::GPU: return std::forward<GpuFn>(gpuFn)();
            default: throw std::runtime_error(std::string(fnName) + ": Unknown Device");
        }
    }

    template <typename CpuFn, typename GpuFn>   //&& means universal reference (includes returnvalues etc)
    decltype(auto) deviceDispatcher(const char* fnName , CpuFn&& cpuFn, GpuFn&& gpuFn) const {
        switch(storage -> device.type) {
            case DeviceType::CPU: return std::forward<CpuFn>(cpuFn)();
            case DeviceType::GPU: return std::forward<GpuFn>(gpuFn)();
            default: throw std::runtime_error(std::string(fnName) + ": Unknown Device");
        }
    }
};

}   //namespace alya::internal