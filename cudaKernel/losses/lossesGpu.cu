#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <cstddef>
#include <string>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/losses/CeSo.hpp>
#include <alya/losses/Bce.hpp>
#include <alya/losses/Mse.hpp>
#include <alya/losses/Mae.hpp>
#include <alya/losses/KlDivergence.hpp>
#include <alya/losses/Huber.hpp>
#include <alya/losses/Hinge.hpp>
#include <alya/losses/CosineSimilarity.hpp>


template <typename T>
__global__ void ceSoKernel(const T* logits, const T* targets, T* grad, T* lossPartial, size_t classes) {
    extern __shared__ unsigned char shmem[];

    T* shLogits = reinterpret_cast<T*>(shmem);
    T* shExp = shLogits + classes;

    size_t row = blockIdx.x;
    size_t LOCAL_THREAD_X = threadIdx.x;

    logits += row * classes;
    targets += row * classes;
    grad += row * classes;

    if(LOCAL_THREAD_X < classes) {
        shLogits[LOCAL_THREAD_X] = logits[LOCAL_THREAD_X];
    }

    __syncthreads();

    T maxLogit = shLogits[0];

    for(size_t i = 1; i < classes; i++) {
        maxLogit = gpuMax(maxLogit, shLogits[i]);
    }

    if(LOCAL_THREAD_X < classes) {
        shExp[LOCAL_THREAD_X] = exp(shLogits[LOCAL_THREAD_X] - maxLogit);
    }

    __shared__ T shSum;

    if(LOCAL_THREAD_X == 0) {
        T sum = gpuZero<T>();
        for(size_t i = 0; i < classes; i++) {
            sum += shExp[i];
        }

        shSum = sum;
    }

    __syncthreads();

    if(LOCAL_THREAD_X < classes) {
        T softmax = shExp[LOCAL_THREAD_X] / shSum;
        T t = targets[LOCAL_THREAD_X];

        grad[LOCAL_THREAD_X] = softmax - t;
    }

    if(LOCAL_THREAD_X == 0) {
        T loss = gpuZero<T>();
        for(size_t i = 0; i < classes; i++) {
            if(targets[i] == gpuOne<T>()) {
                T softmax = shExp[i] / shSum;
                loss = -gpuLog(softmax + toStorage<T>(1e-6));
                break;
            }
        }

        lossPartial[row] = loss;
    }
}

template <typename T>
__global__ void cosineSimilarityKernel(const T* logits, const T* targets, T* grad, T* lossPartial, T eps, size_t rows, size_t cols) {
    extern __shared__ unsigned char shmem_bytes[];
    T* shmem = reinterpret_cast<T*>(shmem_bytes);

    T* dotS = shmem;
    T* normPS = shmem + blockDim.x;
    T* normTS = shmem + 2 * blockDim.x;

    size_t row = blockIdx.x;
    size_t j = threadIdx.x;

    if (row >= rows) return;

    size_t idx = row * cols + j;

    T p = (j < cols) ? logits[idx] : gpuZero<T>();
    T t = (j < cols) ? targets[idx] : gpuZero<T>();

    T pdot = p * t;
    T pp   = p * p;
    T tt   = t * t;

    dotS[j] = pdot;
    normPS[j] = pp;
    normTS[j] = tt;

    __syncthreads();

    for (int stride = cols / 2; stride > 0; stride >>= 1) {
        if (j < stride) {
            dotS[j] += dotS[j + stride];
            normPS[j] += normPS[j + stride];
            normTS[j] += normTS[j + stride];
        }
        __syncthreads();
    }

    if (j == 0) {
        T dot = dotS[0];
        T norm_p = gpuSqrt(normPS[0]);
        T norm_t = gpuSqrt(normTS[0]);

        norm_p = gpuMax(norm_p, eps);
        norm_t = gpuMax(norm_t, eps);

        T cos_sim = dot / (norm_p * norm_t);

        T loss = gpuOne<T>() - cos_sim;
        atomicAdd(lossPartial, loss);
    }

    __syncthreads();

    T dot = dotS[0];
    T norm_p = gpuSqrt(normPS[0]);
    T norm_t = gpuSqrt(normTS[0]);

    norm_p = gpuMax(norm_p, eps);
    norm_t = gpuMax(norm_t, eps);

    T denom = norm_p * norm_t;

    if (j < cols) {
        T p_val = logits[idx];
        T t_val = targets[idx];

        grad[idx] =(t_val / denom - (dot / (norm_p * norm_p * norm_p * norm_t)) * p_val);
    }
}

template <typename T>
__global__ void hingeKernel(const T* logits, const T* targets, T* grad, T* lossPartial, size_t N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX >= N) return;

    T p = logits[GLOBAL_IDX];
    T t = targets[GLOBAL_IDX];
    T loss = gpuZero<T>();

    T diff = gpuOne<T>() - p * t;

    if(diff > gpuZero<T>()) {
        loss += diff;
        grad[GLOBAL_IDX] = -t / toStorage<T>(N);
    } else {
        grad[GLOBAL_IDX] = gpuZero<T>();
    }

    atomicAdd(lossPartial, loss);
}

template <typename T>
__global__ void huberKernel(const T* logits, const T* targets, T* grad, T* lossPartial, T delta, size_t N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX >= N) return;

    T diff = logits[GLOBAL_IDX] - targets[GLOBAL_IDX];
    T loss = gpuZero<T>();

    if(gpuAbs<T>(diff) <= delta) {
        loss += toStorage<T>(0.5) * diff * diff;
        grad[GLOBAL_IDX] = diff / toStorage<T>(N);
    } else {
        loss += delta * (gpuAbs<T>(diff) - toStorage<T>(0.5) * delta);
        grad[GLOBAL_IDX] = (diff > gpuZero<T>() ? delta : -delta) / toStorage<T>(N);
    }

    atomicAdd(lossPartial, loss);
}

template <typename T>
__global__ void klDivergenceKernel(const T* logits, const T* targets, T* grad, T* lossPartial, size_t classes) {
    extern __shared__ unsigned char shmem[];

    T* shLogits = reinterpret_cast<T*>(shmem);
    T* shExp = reinterpret_cast<T*>(shmem + classes * sizeof(T));

    size_t row = blockIdx.x;
    size_t LOCAL_THREAD_X = threadIdx.x;

    logits += row * classes;
    targets += row * classes;
    grad += row * classes;

    if(LOCAL_THREAD_X < classes) {
        shLogits[LOCAL_THREAD_X] = logits[LOCAL_THREAD_X];
    }

    __syncthreads();

    T maxLogit = shLogits[0];
    for(size_t i = 1; i < classes; i++) {
        maxLogit = gpuMax(maxLogit, shLogits[i]);
    }

    if(LOCAL_THREAD_X < classes) {
        shExp[LOCAL_THREAD_X] = exp(shLogits[LOCAL_THREAD_X] - maxLogit);
    }

    __shared__ T sumExp;
    if(LOCAL_THREAD_X == 0) {
        T sum = gpuZero<T>();
        for(size_t i = 0; i < classes; i++) {
            sum += shExp[i];
        }

        sumExp = sum;
    }

    __syncthreads();

    if(LOCAL_THREAD_X < classes) {
        T softmax = shExp[LOCAL_THREAD_X] / sumExp;
        grad[LOCAL_THREAD_X] = softmax - targets[LOCAL_THREAD_X];
        
        const T eps = toStorage<T>(1e-6);
        lossPartial[row] = gpuZero<T>();
        if(targets[LOCAL_THREAD_X] > gpuZero<T>()) {
            atomicAdd(&lossPartial[row], targets[LOCAL_THREAD_X] * gpuLog((targets[LOCAL_THREAD_X] + eps) / (softmax + eps)));
        }
    }
}

template <typename T>
__global__ void bceKernel(const T* logits, const T* targets, T* grad, T* lossPartial, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX >= N) return;

    T x = logits[GLOBAL_IDX];
    T y = targets[GLOBAL_IDX];

    T sigmoid;
    if(x >= gpuZero<T>()) {
        T z = exp(-x); 
        sigmoid = gpuOne<T>() / (gpuOne<T>() + z);
    } else {
        T z = exp(x);
        sigmoid = z / (gpuOne<T>() + z);
    }

    grad[GLOBAL_IDX] = sigmoid - y;

    const T epsylon = toStorage<T>(1e-6);
    lossPartial[GLOBAL_IDX] = - (y * gpuLog(sigmoid + epsylon) + (gpuOne<T>() - y) * gpuLog(gpuOne<T>() - sigmoid + epsylon));
}

template <typename T>
__global__ void mseKernel(const T* logits, const T* targets, T* grad, T* lossPartial, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX >= N) return;

    T differnce = logits[GLOBAL_IDX] - targets[GLOBAL_IDX];
    grad[GLOBAL_IDX] = toStorage<T>(2) * differnce / toStorage<T>(N);

    lossPartial[GLOBAL_IDX] = differnce * differnce;
}

template <typename T>
__global__ void maeKernel(const T* logits, const T* targets, T* grad, T* lossPartial, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;

    if(GLOBAL_IDX >= N) return;

    T differnce = logits[GLOBAL_IDX] - targets[GLOBAL_IDX];
    grad[GLOBAL_IDX] = (differnce > gpuZero<T>() ? gpuOne<T>() : (differnce < gpuZero<T>() ? -gpuOne<T>() : gpuZero<T>())) / toStorage<T>(N);

    lossPartial[GLOBAL_IDX] = gpuAbs(differnce);
}

template <typename T>
__global__ void reduceLossKernel(T* partial, int N) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int BLOCK_SIZE_X = blockDim.x;
    const int GLOBAL_THREAD_X = GLOBAL_BLOCK_X * BLOCK_SIZE_X + LOCAL_THREAD_X;

    shmem[LOCAL_THREAD_X] = (GLOBAL_THREAD_X < N) ? partial[GLOBAL_THREAD_X] : gpuZero<T>();

    __syncthreads();

    for(int reduct_idx = BLOCK_SIZE_X / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            shmem[LOCAL_THREAD_X] += shmem[LOCAL_THREAD_X + reduct_idx];
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuZero<T>();

    if(LOCAL_THREAD_X < 32) {
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }

    if(LOCAL_THREAD_X == 0) {
        partial[GLOBAL_BLOCK_X] = val;
    }
}

namespace alya {

template <typename P>
internal::lossResult<P> CrossEntropyLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();

    const_cast<Tensor<P, 2>&>(logits).toGPU();
    const_cast<Tensor<P, 2>&>(targets).toGPU();

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, batch * sizeof(storageT));

    const dim3 blockSize(classes);
    const dim3 gridSize(batch);
    const size_t shmemSize = 2 * classes * sizeof(storageT);


    ceSoKernel<storageT><<<gridSize, blockSize, shmemSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, classes);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: CrossEntropyLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = batch;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;

        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));

        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> cosineSimilarityLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();

    const_cast<Tensor<P, 2>&>(logits).toGPU();
    const_cast<Tensor<P, 2>&>(targets).toGPU();

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    storageT eps = toStorage<storageT>(1e-6);

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, batch * sizeof(storageT));

    const dim3 blockSize(batch);
    const dim3 gridSize(classes);
    const size_t shmemSize = 2 * classes * sizeof(storageT);


    cosineSimilarityKernel<storageT><<<blockSize, gridSize, shmemSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, eps, batch, classes);
    cudaError_t err = cudaDeviceSynchronize();
    
    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: cosineSimilarityLos: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = batch;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> hingeLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();
    const size_t N = classes * batch;

    const_cast<Tensor<P, 2>&>(logits).toGPU();
    const_cast<Tensor<P, 2>&>(targets).toGPU();

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, batch * sizeof(storageT));

    const dim3 blockSize(batch);
    const dim3 gridSize(classes);
    const size_t shmemSize = 2 * classes * sizeof(storageT);


    hingeKernel<storageT><<<blockSize, gridSize, shmemSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: hingeLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = batch;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);
    
    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> huberLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();
    const size_t N = classes * batch;

    const_cast<Tensor<P, 2>&>(logits).toGPU();
    const_cast<Tensor<P, 2>&>(targets).toGPU();

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT delta = huberLoss<P>::delta;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, batch * sizeof(storageT));

    const dim3 blockSize(batch);
    const dim3 gridSize(classes);
    const size_t shmemSize = 2 * classes * sizeof(storageT);


    huberKernel<storageT><<<blockSize, gridSize, shmemSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, delta, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: huberLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = batch;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> klDivergenceLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();

    const_cast<Tensor<P, 2>&>(logits).toGPU();
    const_cast<Tensor<P, 2>&>(targets).toGPU();

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, batch * sizeof(storageT));

    const dim3 blockSize(batch);
    const dim3 gridSize(classes);
    const size_t shmemSize = 2 * classes * sizeof(storageT);


    klDivergenceKernel<storageT><<<blockSize, gridSize, shmemSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, classes);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: klDivergenceLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = batch;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> bceLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();
    const size_t N = batch * classes;

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, N * sizeof(storageT));

    constexpr int blockSize = 256;
    const int gridSize = ((N + blockSize - 1) / blockSize);

    bceKernel<storageT><<<blockSize, gridSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: bceLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = N;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> mseLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();
    const size_t N = batch * classes;

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, N * sizeof(storageT));

    constexpr int blockSize = 256;
    const int gridSize = ((N + blockSize - 1) / blockSize);

    mseKernel<storageT><<<blockSize, gridSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: mseLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = N;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template <typename P>
internal::lossResult<P> maeLoss<P>::forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
    const size_t batch = logits.numRows();
    const size_t classes = logits.numCols();
    const size_t N = batch * classes;

    Tensor<P, 2> grad(batch, classes, Device{DeviceType::GPU});
    grad.toGPU();

    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    const storageT* d_logits = logits.gpuData();
    const storageT* d_targets = targets.gpuData();
    storageT* d_gradients = grad.gpuData();

    storageT* d_lossPartial;
    cudaMalloc(&d_lossPartial, N * sizeof(storageT));

    constexpr int blockSize = 256;
    const int gridSize = ((N + blockSize - 1) / blockSize);

    maeKernel<storageT><<<blockSize, gridSize>>>(d_logits, d_targets, d_gradients, d_lossPartial, N);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: maeLoss: " + std::string(cudaGetErrorString(err)));
    }

    size_t current = N;
    while(current > 1) {
        constexpr size_t localBlockSize = 256;
        const size_t localGridSize = (current + localBlockSize - 1) / localBlockSize;
        reduceLossKernel<storageT><<<localGridSize, localBlockSize, localBlockSize * sizeof(storageT)>>>(d_lossPartial, static_cast<int>(current));
        current = localGridSize;
    }

    storageT loss;
    cudaMemcpy(&loss, d_lossPartial, sizeof(storageT), cudaMemcpyDeviceToHost);
    cudaFree(d_lossPartial);

    computeT lossVal = toCompute(loss);
    lossVal /= static_cast<computeT>(batch);

    return { lossVal, grad };
}

template class CrossEntropyLoss<bf16>;
template class CrossEntropyLoss<fp16>;
template class CrossEntropyLoss<fp32>;
template class CrossEntropyLoss<fp64>;

template class cosineSimilarityLoss<bf16>;
template class cosineSimilarityLoss<fp16>;
template class cosineSimilarityLoss<fp32>;
template class cosineSimilarityLoss<fp64>;

template class hingeLoss<bf16>;
template class hingeLoss<fp16>;
template class hingeLoss<fp32>;
template class hingeLoss<fp64>;

template class huberLoss<bf16>;
template class huberLoss<fp16>;
template class huberLoss<fp32>;
template class huberLoss<fp64>;

template class klDivergenceLoss<bf16>;
template class klDivergenceLoss<fp16>;
template class klDivergenceLoss<fp32>;
template class klDivergenceLoss<fp64>;

template class bceLoss<bf16>;
template class bceLoss<fp16>;
template class bceLoss<fp32>;
template class bceLoss<fp64>;

template class mseLoss<bf16>;
template class mseLoss<fp16>;
template class mseLoss<fp32>;
template class mseLoss<fp64>;

template class maeLoss<bf16>;
template class maeLoss<fp16>;
template class maeLoss<fp32>;
template class maeLoss<fp64>;

}   //namespace alya