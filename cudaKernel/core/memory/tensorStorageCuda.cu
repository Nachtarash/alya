#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include <alya/core/memory/TensorStorageCuda.cuh>
#include <alya/core/memory/Storage.hpp>
#include <alya/profiling/CudaCheck.cuh>
#include <alya/profiling/ErrorMessage.hpp>

namespace alya::internal {

void tensorStorageAllocateGpu(Storage* storage) {
    if(!storage) {
        ERRORMESSAGE("has nothing to store");
    }

    if(storage -> bytes == 0 || storage -> gpuPtr) {
        return;
    }

    CUDA_CHECK(cudaMalloc(&storage -> gpuPtr, storage -> bytes));
}

void tensorStorageFreeGpu(Storage* storage) noexcept {
    if(!storage || !storage -> gpuPtr) {
        return;
    }

    CUDA_CHECK(cudaFree(storage -> gpuPtr));
    storage -> gpuPtr = nullptr;
}

void tensorStorageSyncToGpu(Storage* storage) {
    if(!storage) {
        ERRORMESSAGE("has nothing to store");
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
        CUDA_CHECK(cudaMemcpy(storage -> gpuPtr, storage -> cpuPtr, storage -> bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(storage -> gpuPtr, 0, storage -> bytes));
    }

    storage -> gpuValid = true;
}

void tensorStorageSyncToCpu(Storage* storage) {
    if(!storage) {
        ERRORMESSAGE("has nothing to store");
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
            ERRORMESSAGE("malloc failed");
        }
    }

    if(storage -> gpuValid) {
        CUDA_CHECK(cudaMemcpy(storage -> cpuPtr, storage -> gpuPtr, storage -> bytes, cudaMemcpyDeviceToHost));
    } else {
        std::memset(storage -> cpuPtr, 0, storage->bytes);
    }

    storage -> cpuValid = true;
}

}   //namespace alya::internal
