#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <string>
#include <stdexcept>

#include <alya/activation/ActivationGpu.cuh>
#include <alya/profiling/CudaCheck.cuh>

//------------- KERNEL -----------------
template <typename Op, typename T>
__global__ void activationKernel(const T* in, T* out, const size_t N) {
    int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX < N) {
        out[GLOBAL_IDX] = Op::apply(in[GLOBAL_IDX]);
    }
}

template <typename Op, typename T>
__global__ void backwardKernel(const T* gradOut, const T* z, const T* a, T* gradZ, const size_t N) {
    int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX < N) {
        gradZ[GLOBAL_IDX] = Op::backwardScalar(gradOut[GLOBAL_IDX], z[GLOBAL_IDX], a[GLOBAL_IDX]);
    }
}

//------------ FUNCTIONS --------------
template <typename Op, typename T>
void activationGpu::applyGpu(const T* in, T* out, const size_t N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    activationKernel<Op, T><<<gridSize, blockSize>>>(in, out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Op, typename T>
void activationGpu::backwardGpu(const T* gradOut, const T* z, const T* a, T* gradZ, const size_t N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    backwardKernel<Op, T><<<gridSize, blockSize>>>(gradOut, z, a, gradZ, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

//----------- CUDA TEMPLATE INSTANTIATIONS -------------

template void activationGpu::applyGpu<LinearOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<LinearOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<LinearOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<LinearOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<LinearOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<LinearOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<LinearOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<LinearOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<ReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<ReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<ReLuOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<ReLuOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<ReLuOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<ReLuOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<ReLuOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<ReLuOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<LeakyReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<LeakyReLuOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<LeakyReLuOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<LeakyReLuOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<LeakyReLuOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<LeakyReLuOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<LeakyReLuOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<LeakyReLuOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<ELUOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<ELUOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<ELUOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<ELUOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<ELUOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<ELUOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<ELUOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<ELUOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<SigmoidOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<SigmoidOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<SigmoidOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<SigmoidOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<SigmoidOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<SigmoidOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<SigmoidOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<SigmoidOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<TanhOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<TanhOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<TanhOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<TanhOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<TanhOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<TanhOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<TanhOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<TanhOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<GELUOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<GELUOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<GELUOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<GELUOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<GELUOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<GELUOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<GELUOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<GELUOp<double>, double>(const double*, const double*, const double*, double*, size_t);

template void activationGpu::applyGpu<SwishOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::backwardGpu<SwishOp<__nv_bfloat16>, __nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, size_t);
template void activationGpu::applyGpu<SwishOp<__half>, __half>(const __half*, __half*, size_t);
template void activationGpu::backwardGpu<SwishOp<__half>, __half>(const __half*, const __half*, const __half*, __half*, size_t);
template void activationGpu::applyGpu<SwishOp<float>, float>(const float*, float*, size_t);
template void activationGpu::backwardGpu<SwishOp<float>, float>(const float*, const float*, const float*, float*, size_t);
template void activationGpu::applyGpu<SwishOp<double>, double>(const double*, double*, size_t);
template void activationGpu::backwardGpu<SwishOp<double>, double>(const double*, const double*, const double*, double*, size_t);
