#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <string>
#include <stdexcept>

#include <alya/activation/ActivationGpu.cuh>

//------------- KERNEL -----------------
template <typename Op, typename T>
__global__ void activationKernel(const T* in, T* out, size_t N) {
    int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX < N) {
        out[GLOBAL_IDX] = Op::apply(in[GLOBAL_IDX]);
    }
}

template <typename Op, typename T>
__global__ void derivativeKernel(const T* y, T* out, size_t N) {
    int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX < N) {
        out[GLOBAL_IDX] = Op::derivativeFromOutput(y[GLOBAL_IDX]);
    }
}

//------------ FUNCTIONS --------------
template <typename Op, typename T>
void activationGpu::applyGpu(const T* in, T* out, size_t N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    activationKernel<Op, T><<<gridSize, blockSize>>>(in, out, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: applyGpu " + std::string(cudaGetErrorString(err)));
    }
}

template <typename Op, typename T>
void activationGpu::derivativeGpu(const T* y, T* out, size_t N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    derivativeKernel<Op, T><<<gridSize, blockSize>>>(y, out, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: derivativeGpu: " + std::string(cudaGetErrorString(err)));
    }
}

//----------- CUDA TEMPLATE INSTANTIATIONS -------------

template void activationGpu::applyGpu<LinearOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::derivativeGpu<LinearOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<LinearOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::derivativeGpu<LinearOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::applyGpu<LinearOp<float>, float>(const float*, float*, size_t);
template void activationGpu::derivativeGpu<LinearOp<float>, float>(const float*, float*, size_t);
template void activationGpu::applyGpu<LinearOp<double>, double>(const double*, double*, size_t);
template void activationGpu::derivativeGpu<LinearOp<double>, double>(const double*, double*, size_t);

template void activationGpu::applyGpu<ReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::derivativeGpu<ReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<ReLuOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::derivativeGpu<ReLuOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::applyGpu<ReLuOp<float>, float>(const float*, float*, size_t);
template void activationGpu::derivativeGpu<ReLuOp<float>, float>(const float*, float*, size_t);
template void activationGpu::applyGpu<ReLuOp<double>, double>(const double*, double*, size_t);
template void activationGpu::derivativeGpu<ReLuOp<double>, double>(const double*, double*, size_t);

template void activationGpu::applyGpu<LeakyReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::derivativeGpu<LeakyReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<LeakyReLuOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::derivativeGpu<LeakyReLuOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::applyGpu<LeakyReLuOp<float>, float>(const float*, float*, size_t);
template void activationGpu::derivativeGpu<LeakyReLuOp<float>, float>(const float*, float*, size_t);
template void activationGpu::applyGpu<LeakyReLuOp<double>, double>(const double*, double*, size_t);
template void activationGpu::derivativeGpu<LeakyReLuOp<double>, double>(const double*, double*, size_t);

template void activationGpu::applyGpu<ELUOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::derivativeGpu<ELUOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<ELUOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::derivativeGpu<ELUOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::applyGpu<ELUOp<float>, float>(const float*, float*, size_t);
template void activationGpu::derivativeGpu<ELUOp<float>, float>(const float*, float*, size_t);
template void activationGpu::applyGpu<ELUOp<double>, double>(const double*, double*, size_t);
template void activationGpu::derivativeGpu<ELUOp<double>, double>(const double*, double*, size_t);

template void activationGpu::applyGpu<SigmoidOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::derivativeGpu<SigmoidOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<SigmoidOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::derivativeGpu<SigmoidOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::applyGpu<SigmoidOp<float>, float>(const float*, float*, size_t);
template void activationGpu::derivativeGpu<SigmoidOp<float>, float>(const float*, float*, size_t);
template void activationGpu::applyGpu<SigmoidOp<double>, double>(const double*, double*, size_t);
template void activationGpu::derivativeGpu<SigmoidOp<double>, double>(const double*, double*, size_t);

template void activationGpu::applyGpu<TanhOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::derivativeGpu<TanhOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<TanhOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::derivativeGpu<TanhOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::applyGpu<TanhOp<float>, float>(const float*, float*, size_t);
template void activationGpu::derivativeGpu<TanhOp<float>, float>(const float*, float*, size_t);
template void activationGpu::applyGpu<TanhOp<double>, double>(const double*, double*, size_t);
template void activationGpu::derivativeGpu<TanhOp<double>, double>(const double*, double*, size_t);
