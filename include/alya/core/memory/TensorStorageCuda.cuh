#pragma once

#include <cuda_runtime.h>

#include <alya/core/memory/Device.hpp>
#include <alya/core/memory/Storage.hpp>
#include <alya/profiling/CudaCheck.cuh>
#include <alya/profiling/ErrorMessage.hpp>

namespace alya::internal {

void tensorStorageAllocateGpu(Storage* storage);
void tensorStorageFreeGpu(Storage* storage) noexcept;
void tensorStorageSyncToGpu(Storage* storage);
void tensorStorageSyncToCpu(Storage* storage);

template <typename TensorType>
TensorType TensorCloneCuda(const TensorType& src, TensorType out) {
    if(!src.storage) { return out; }

    if(src.storage -> cpuValid) {
        if(out.storage -> device.type == DeviceType::GPU) {
            if(!out.storage -> gpuPtr) {
                CUDA_CHECK(cudaMalloc(&out.storage -> gpuPtr, out.storage -> bytes));
            }

            cudaMemcpy(out.storage -> gpuPtr, src.storage -> cpuPtr, src.storage -> bytes, cudaMemcpyHostToDevice);

            out.storage -> gpuValid = true;
            out.storage -> cpuValid = false;

            return out;
        }

        if(!out.storage -> cpuPtr) {
            out.storage -> cpuPtr = std::malloc(out.storage -> bytes);

            if(!out.storage -> cpuPtr) {
                ERRORMESSAGE("malloc failed");
            }
        }

        std::memcpy(out.storage -> cpuPtr, src.storage -> cpuPtr, src.storage -> bytes);

        out.storage -> cpuValid = true;
        out.storage -> gpuValid = false;

        return out;
    }

    if(src.storage -> gpuValid) {
        if(out.storage -> device.type == DeviceType::GPU) {
            if(!out.storage -> gpuPtr) {
                CUDA_CHECK(cudaMalloc(&out.storage -> gpuPtr, out.storage -> bytes));
            }

            cudaMemcpy(out.storage -> gpuPtr, src.storage -> gpuPtr, src.storage -> bytes, cudaMemcpyDeviceToDevice);

            out.storage -> gpuValid = true;
            out.storage -> cpuValid = false;

            return out;
        }

        if(!out.storage -> cpuPtr) {
            out.storage -> cpuPtr = std::malloc(out.storage -> bytes);

            if(!out.storage -> cpuPtr) {
                ERRORMESSAGE("malloc failed");
            }
        }

        cudaMemcpy(out.storage -> cpuPtr, src.storage -> cpuPtr, src.storage -> bytes, cudaMemcpyDeviceToHost);

        out.storage -> cpuValid = true;
        out.storage -> gpuValid = false;

        return out;
    }

    return out;
}

}   //namespace alya::internal
