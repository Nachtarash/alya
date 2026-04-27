#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <alya/core/memory/Storage.hpp>

namespace alya::internal {

void tensorStorageAllocateGpu(Storage* storage) {
    if(!storage) {
        throw std::runtime_error("GPU: tensorStorageAllocateGpu: has nothing to store");
    }

    if(storage -> bytes == 0 || storage -> gpuPtr) {
        return;
    }

    cudaError_t err = cudaMalloc(&storage -> gpuPtr, storage -> bytes);
    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: tensorStorageAllocateGpu: cudamalloc failed: " + std::string(cudaGetErrorString(err)));
    }
}

void tensorStorageFreeGpu(Storage* storage) noexcept {
    if(!storage || !storage -> gpuPtr) {
        return;
    }

    cudaFree(storage -> gpuPtr);
    storage -> gpuPtr = nullptr;
}

void tensorStorageSyncToGpu(Storage* storage) {
    if(!storage) {
        throw std::runtime_error("GPU: tensorStorageSyncToGpu: failed: has nothing to store");
    }

    if(storage -> bytes == 0) {
        storage -> gpuValid = true;
        return;
    }

    if(storage -> gpuValid) {
        return;
    }

    tensorStorageAllocateGpu(storage);

    if(storage -> cpuValid) {
        cudaError_t err = cudaMemcpy(storage -> gpuPtr, storage -> cpuPtr, storage -> bytes, cudaMemcpyHostToDevice);
        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: tensorStorageSyncToGpu: cudaMemcpy H2D failed: " + std::string(cudaGetErrorString(err)));
        }
    } else {
        cudaError_t err = cudaMemset(storage -> gpuPtr, 0, storage -> bytes);
        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: tensorStorageSyncToGpu: cudaMemset failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    storage -> gpuValid = true;
}

void tensorStorageSyncToCpu(Storage* storage) {
    if(!storage) {
        throw std::runtime_error("CPU: tensorStorageSyncToCpu: has nothing to store");
    }

    if(storage -> bytes == 0) {
        storage -> cpuValid = true;
        return;
    }

    if(storage -> cpuValid) {
        return;
    }

    if(!storage -> cpuPtr) {
        storage -> cpuPtr = std::malloc(storage -> bytes);
        if(!storage -> cpuPtr) {
            throw std::runtime_error("CPU: tensorStorageSyncToCpu: malloc failed");
        }
    }

    if(storage -> gpuValid) {
        cudaError_t err = cudaMemcpy(storage -> cpuPtr, storage -> gpuPtr, storage -> bytes, cudaMemcpyDeviceToHost);
        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: tensorStorageSyncToCpu: cudaMemcpy D2H failed: " + std::string(cudaGetErrorString(err)));
        }
    } else {
        std::memset(storage -> cpuPtr, 0, storage->bytes);
    }

    storage -> cpuValid = true;
}

}   //namespace alya::internal