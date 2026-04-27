#include <cuda_runtime.h>

#include <cassert>
#include <string>
#include <cstdlib>
#include <cstddef>
#include <stdexcept>

#include <alya/core/memory/Device.hpp>
#include <alya/core/memory/Storage.hpp>
#include <alya/core/memory/TensorStorageCuda.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/tensor/Tensoraxis.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/core/precision/NumericLimits.cuh>
#include <alya/core/ops/NumericalOps.cuh>
#include <alya/core/ops/WarpReduction.hpp>
#include <alya/core/data/ArgMinMaxContainer.hpp>


//--------------- KERNEL---------------//

namespace MatmulConfig {
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_N = 8;
    constexpr int MATMUL_VALUES_K = 8;
    constexpr int THREAD_TILE_SIZE = 2;
};

//2x2 values per thread getting calculated, B gets loaded transposed + coalesced memory --> for warp efficiency
template <typename T>
__global__ void matmul2x2Kernel(const T* A, const T* B, T* C, int M, int K, int N) {
    const int THREAD_VALUES_M = MatmulConfig::BLOCK_M * MatmulConfig::THREAD_TILE_SIZE;
    const int THREAD_VALUES_N = MatmulConfig::BLOCK_N * MatmulConfig::THREAD_TILE_SIZE;

    const int LOCAL_THREAD_ROW = threadIdx.y;
    const int LOCAL_THREAD_COL = threadIdx.x;

    const int GLOBAL_BLOCK_ROW = blockIdx.y * THREAD_VALUES_M + LOCAL_THREAD_ROW * MatmulConfig::THREAD_TILE_SIZE;
    const int GLOBAL_BLOCK_COL = blockIdx.x * THREAD_VALUES_N + LOCAL_THREAD_COL * MatmulConfig::THREAD_TILE_SIZE;

    __shared__ T As[THREAD_VALUES_M][MatmulConfig::MATMUL_VALUES_K + 1];
    __shared__ T Bs[THREAD_VALUES_N][MatmulConfig::MATMUL_VALUES_K + 1];

    T v00 = gpuZero<T>();
    T v01 = gpuZero<T>();
    T v10 = gpuZero<T>();
    T v11 = gpuZero<T>();

    const int K_TILES = (K + MatmulConfig::MATMUL_VALUES_K - 1) / MatmulConfig::MATMUL_VALUES_K;

    for(int k_tile = 0; k_tile < K_TILES; k_tile++) {
        const int K_TILE_IDX = k_tile * MatmulConfig::MATMUL_VALUES_K;

        const int A_ROW_0 = GLOBAL_BLOCK_ROW;
        const int A_ROW_1 = GLOBAL_BLOCK_ROW + 1;
        const int A_COL = LOCAL_THREAD_COL + K_TILE_IDX;

        const int B_COL_0 = GLOBAL_BLOCK_COL;
        const int B_COL_1 = GLOBAL_BLOCK_COL + 1;
        const int B_ROW = LOCAL_THREAD_ROW + K_TILE_IDX;
    
        As[LOCAL_THREAD_ROW * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_COL] = (A_ROW_0 < M && A_COL < K) ? A[A_ROW_0 * K + A_COL] : gpuZero<T>();
        As[LOCAL_THREAD_ROW * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_COL] = (A_ROW_1 < M && A_COL < K) ? A[A_ROW_1 * K + A_COL] : gpuZero<T>();
                                                                                                                     
        Bs[LOCAL_THREAD_COL * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_ROW] = (B_ROW < K && B_COL_0 < N) ? B[B_ROW * N + B_COL_0] : gpuZero<T>();       
        Bs[LOCAL_THREAD_COL * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_ROW] = (B_ROW < K && B_COL_1 < N) ? B[B_ROW * N + B_COL_1] : gpuZero<T>();

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < MatmulConfig::MATMUL_VALUES_K; k++) {
            T temp_a0 = As[LOCAL_THREAD_ROW * MatmulConfig::THREAD_TILE_SIZE][k];        
            T temp_a1 = As[LOCAL_THREAD_ROW * MatmulConfig::THREAD_TILE_SIZE + 1][k];

            T temp_b0 = Bs[LOCAL_THREAD_COL * MatmulConfig::THREAD_TILE_SIZE][k]; 
            T temp_b1 = Bs[LOCAL_THREAD_COL * MatmulConfig::THREAD_TILE_SIZE + 1][k];

            v00 += temp_a0 * temp_b0;
            v01 += temp_a0 * temp_b1;
            v10 += temp_a1 * temp_b0;
            v11 += temp_a1 * temp_b1;
        }

        __syncthreads();
    }

    if(GLOBAL_BLOCK_ROW < M && GLOBAL_BLOCK_COL < N) {
        C[GLOBAL_BLOCK_ROW * N + GLOBAL_BLOCK_COL] = v00;
    }

    if(GLOBAL_BLOCK_ROW < M && GLOBAL_BLOCK_COL + 1 < N) {
        C[GLOBAL_BLOCK_ROW * N + GLOBAL_BLOCK_COL + 1] = v01;
    }
    
    if(GLOBAL_BLOCK_ROW + 1 < M && GLOBAL_BLOCK_COL < N) {
        C[(GLOBAL_BLOCK_ROW + 1) * N + GLOBAL_BLOCK_COL] = v10;
    }
    
    if(GLOBAL_BLOCK_ROW + 1 < M && GLOBAL_BLOCK_COL + 1 < N) {
        C[(GLOBAL_BLOCK_ROW + 1) * N + GLOBAL_BLOCK_COL + 1] = v11;
    }
}

//Addition/Multiplication only on flattend j index -> for bias
template <typename Op, typename T, TensorAxis axis>
__global__ void broadcastKernel(const T* A, const T* B, T* C, int rows, int cols) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < rows * cols) {
        [[maybe_unused]] const int r = GLOBAL_IDX / cols;
        [[maybe_unused]] const int c = GLOBAL_IDX % cols;

        if constexpr(axis == TensorAxis::Row) {
            C[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], B[c]); //ROW
        } else {
            C[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], B[r]); //COL
        }
    }
}

template <typename Op, typename T, TensorAxis axis>
__global__ void broadcast_inplaceKernel(T* A, const T* B, int rows, int cols) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < rows * cols) {
        [[maybe_unused]] const int r = GLOBAL_IDX / cols;
        [[maybe_unused]] const int c = GLOBAL_IDX % cols;

        if constexpr(axis == TensorAxis::Row) {
            A[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], B[c]); //ROW
        } else {
            A[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], B[r]); //COL
        }
    }
}

namespace TransposeConfig {
    constexpr int TILE_SIZE = 32;
};

template <typename T>
__global__ void transposeKernel(const T* A, T* B, int rows, int cols) {
    __shared__ T shmem[TransposeConfig::TILE_SIZE][TransposeConfig::TILE_SIZE + 1];

    int GLOBAL_X = blockIdx.x * TransposeConfig::TILE_SIZE + threadIdx.x;
    int GLOBAL_Y = blockIdx.y * TransposeConfig::TILE_SIZE + threadIdx.y;
    
    const int THREAD_Y = threadIdx.y;
    const int THREAD_X = threadIdx.x;

    if(GLOBAL_X < cols && GLOBAL_Y < rows) {
        shmem[THREAD_Y][THREAD_X] = A[GLOBAL_Y * cols + GLOBAL_X];
    }

    __syncthreads();

    GLOBAL_X = blockIdx.y * TransposeConfig::TILE_SIZE + threadIdx.x;
    GLOBAL_Y = blockIdx.x * TransposeConfig::TILE_SIZE + threadIdx.y;

    if(GLOBAL_X < rows && GLOBAL_Y < cols) {
        B[GLOBAL_Y * rows + GLOBAL_X] = shmem[THREAD_X][THREAD_Y];
    }
}

template <typename T>
__global__ void sumRowsKernel(const T* A, T* B, int rows, int cols) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int LOCAL_THREAD_X = threadIdx.x;

    T local_thread_sum = gpuZero<T>();

    for(int thread_x = LOCAL_THREAD_X; thread_x < rows; thread_x += blockDim.x) {
        local_thread_sum += A[thread_x * cols + GLOBAL_BLOCK_X];
    }

    shmem[LOCAL_THREAD_X] = local_thread_sum;
    __syncthreads();

    for(int reduct_idx = blockDim.x / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            shmem[LOCAL_THREAD_X] += shmem[LOCAL_THREAD_X + reduct_idx];
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuZero<T>();

    if(LOCAL_THREAD_X < 32) {
        val = warpReductSum(val);
    }

    if(LOCAL_THREAD_X == 0) {
        B[GLOBAL_BLOCK_X] = val;
    }
}

template <typename T>
__global__ void sumColsKernel(const T* A, T* B, int cols) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int LOCAL_THREAD_X = threadIdx.x;

    T local_thread_sum = gpuZero<T>();

    for(int thread_x = LOCAL_THREAD_X; thread_x < cols; thread_x += blockDim.x) {
        local_thread_sum += A[GLOBAL_BLOCK_X * cols + thread_x];
    }

    shmem[LOCAL_THREAD_X] = local_thread_sum;
    __syncthreads();

    for(int reduct_idx = blockDim.x / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            shmem[LOCAL_THREAD_X] += shmem[LOCAL_THREAD_X + reduct_idx];
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuZero<T>();

    if(LOCAL_THREAD_X < 32) {
        val = warpReductSum(val);
    }

    if(LOCAL_THREAD_X == 0) {
        B[GLOBAL_BLOCK_X] = val;
    }
}

template <typename T>
__global__ void minRowsKernel(const T* A, T* B, int rows, int cols) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int LOCAL_THREAD_X = threadIdx.x;

    T local_thread_val = gpuMaxVal<T>();

    for(int thread_x = LOCAL_THREAD_X; thread_x < rows; thread_x += blockDim.x) {
        local_thread_val = gpuMin(local_thread_val, A[thread_x * cols + GLOBAL_BLOCK_X]);
    }

    shmem[LOCAL_THREAD_X] = local_thread_val;
    __syncthreads();

    for(int reduct_idx = blockDim.x / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            shmem[LOCAL_THREAD_X] = gpuMin(shmem[LOCAL_THREAD_X], shmem[LOCAL_THREAD_X + reduct_idx]);
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuMaxVal<T>();

    if(LOCAL_THREAD_X < 32) {
        val = warpReductMin(val);
    }

    if(LOCAL_THREAD_X == 0) {
        B[GLOBAL_BLOCK_X] = val;
    }
}

template <typename T>
__global__ void maxRowsKernel(const T* A, T* B, int rows, int cols) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int LOCAL_THREAD_X = threadIdx.x;

    T local_thread_val = gpuMinVal<T>();

    for(int thread_x = LOCAL_THREAD_X; thread_x < rows; thread_x += blockDim.x) {
        local_thread_val = gpuMax(local_thread_val, A[thread_x * cols + GLOBAL_BLOCK_X]);
    }

    shmem[LOCAL_THREAD_X] = local_thread_val;
    __syncthreads();

    for(int reduct_idx = blockDim.x / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            shmem[LOCAL_THREAD_X] = gpuMax(shmem[LOCAL_THREAD_X], shmem[LOCAL_THREAD_X + reduct_idx]);
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuMinVal<T>();

    if(LOCAL_THREAD_X < 32) {
        val = warpReductMax(val);
    }

    if(LOCAL_THREAD_X == 0) {
        B[GLOBAL_BLOCK_X] = val;
    }
}

template <typename T>
__global__ void argminRowsKernel(const T* A, size_t* B, size_t rows, size_t cols) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const size_t GLOBAL_BLOCK_X = blockIdx.x;
    const size_t LOCAL_THREAD_X = threadIdx.x;

    size_t* shmem_idxs = reinterpret_cast<size_t*>(cache_bytes + blockDim.x * sizeof(T));

    T local_thread_val = gpuMaxVal<T>();
    size_t thread_idx = 0;

    for(size_t thread_x = LOCAL_THREAD_X; thread_x < rows; thread_x += blockDim.x) {
        T value = A[thread_x * cols + GLOBAL_BLOCK_X];

        if(value < local_thread_val || value == local_thread_val && thread_x < thread_idx) {
            local_thread_val = value;
            thread_idx = thread_x;
        }
    }

    shmem[LOCAL_THREAD_X] = local_thread_val;
    shmem_idxs[LOCAL_THREAD_X] = thread_idx;

    __syncthreads();

    for(int reduct_idx = blockDim.x / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            if(shmem[LOCAL_THREAD_X + reduct_idx] < shmem[LOCAL_THREAD_X] || (shmem[LOCAL_THREAD_X + reduct_idx] == shmem[LOCAL_THREAD_X] && shmem_idxs[LOCAL_THREAD_X + reduct_idx] < shmem_idxs[LOCAL_THREAD_X])) {
                shmem[LOCAL_THREAD_X] = shmem[LOCAL_THREAD_X + reduct_idx];
                shmem_idxs[LOCAL_THREAD_X] = shmem_idxs[LOCAL_THREAD_X + reduct_idx];
            }
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuMaxVal<T>();
    size_t idx = (LOCAL_THREAD_X < 32) ? shmem_idxs[LOCAL_THREAD_X] : size_t(0);

    if(LOCAL_THREAD_X < 32) {
        warpReductArgmin(val, idx);
    }

    if(LOCAL_THREAD_X == 0) {
        B[GLOBAL_BLOCK_X] = idx;
    }
}

template <typename T>
__global__ void argmaxRowsKernel(const T* A, size_t* B, size_t rows, size_t cols) {
    extern __shared__ unsigned char cache_bytes[];
    T* shmem = reinterpret_cast<T*>(cache_bytes);

    const size_t GLOBAL_BLOCK_X = blockIdx.x;
    const size_t LOCAL_THREAD_X = threadIdx.x;

    size_t* shmem_idxs = reinterpret_cast<size_t*>(cache_bytes + blockDim.x * sizeof(T));

    T local_thread_val = gpuMinVal<T>();
    size_t thread_idx = 0;

    for(size_t thread_x = LOCAL_THREAD_X; thread_x < rows; thread_x += blockDim.x) {
        T value = A[thread_x * cols + GLOBAL_BLOCK_X];

        if(value > local_thread_val || (value == local_thread_val && thread_x > thread_idx)) {
            local_thread_val = value;
            thread_idx = thread_x;
        }
    }

    shmem[LOCAL_THREAD_X] = local_thread_val;
    shmem_idxs[LOCAL_THREAD_X] = thread_idx;

    __syncthreads();

    for(int reduct_idx = blockDim.x / 2; reduct_idx > 32; reduct_idx >>= 1) {
        if(LOCAL_THREAD_X < reduct_idx) {
            if(shmem[LOCAL_THREAD_X + reduct_idx] > shmem[LOCAL_THREAD_X] || (shmem[LOCAL_THREAD_X + reduct_idx] == shmem[LOCAL_THREAD_X] && shmem_idxs[LOCAL_THREAD_X + reduct_idx] > shmem_idxs[LOCAL_THREAD_X])) {
                shmem[LOCAL_THREAD_X] = shmem[LOCAL_THREAD_X + reduct_idx];
                shmem_idxs[LOCAL_THREAD_X] = shmem_idxs[LOCAL_THREAD_X + reduct_idx];
            }
        }

        __syncthreads();
    }

    T val = (LOCAL_THREAD_X < 32) ? shmem[LOCAL_THREAD_X] : gpuMinVal<T>();
    size_t idx = (LOCAL_THREAD_X < 32) ? shmem_idxs[LOCAL_THREAD_X] : size_t(0);

    if(LOCAL_THREAD_X < 32) {
        warpReductArgmax(val, idx);
    }

    if(LOCAL_THREAD_X == 0) {
        B[GLOBAL_BLOCK_X] = idx;
    }
}

namespace alya {

//--------------- Memory ---------------//
template <typename P>
Tensor<P, 2> Tensor<P, 2>::clone() const {
    Tensor<P, 2> out(rows, cols, storage ? storage -> device : Device{});

    if(!storage) { return out; }

    if(storage -> cpuValid) {
        if(out.storage -> device.type == DeviceType::GPU) {
            if(!out.storage -> gpuPtr) {
                cudaError_t err = cudaMalloc(&out.storage -> gpuPtr, out.storage -> bytes);

                if(err != cudaSuccess) {
                    throw std::runtime_error("GPU: cudamalloc failed: " + std::string(cudaGetErrorString(err)));
                }
            }
            cudaMemcpy(out.storage -> gpuPtr, storage -> cpuPtr, storage -> bytes, cudaMemcpyHostToDevice);

            out.storage -> gpuValid = true;
            out.storage -> cpuValid = false;

            return out;
        }

        if(!out.storage -> cpuPtr) {
            out.storage -> cpuPtr = std::malloc(out.storage -> bytes);

            if(!out.storage -> cpuPtr) {
                throw std::runtime_error("CPU: malloc failed");
            }
        }
        std::memcpy(out.storage -> cpuPtr, storage -> cpuPtr, storage -> bytes);

        out.storage -> cpuValid = true;
        out.storage -> gpuValid = false;

        return out;
    }

    if(storage -> gpuValid) {
        if(out.storage -> device.type == DeviceType::GPU) {
            if(!out.storage -> gpuPtr) {
                cudaError_t err = cudaMalloc(&out.storage -> gpuPtr, out.storage -> bytes);

                if(err != cudaSuccess) {
                    throw std::runtime_error("GPU: cudamalloc failed: " + std::string(cudaGetErrorString(err)));
                }
            }
            cudaMemcpy(out.storage -> gpuPtr, storage -> gpuPtr, storage -> bytes, cudaMemcpyDeviceToDevice);

            out.storage -> gpuValid = true;
            out.storage -> cpuValid = false;

            return out;
        }

        if(!out.storage -> cpuPtr) {
            out.storage -> cpuPtr = std::malloc(out.storage -> bytes);

            if(!out.storage -> cpuPtr) {
                throw std::runtime_error("CPU: malloc failed");
            }
        }
        cudaMemcpy(out.storage -> cpuPtr, storage -> gpuPtr, storage -> bytes, cudaMemcpyDeviceToHost);

        out.storage -> cpuValid = true;
        out.storage -> gpuValid = false;

        return out;
    }

    return out;
}

//--------------- FUNCTIONS ---------------//
//Tensor2D-multiplikation
template <typename P>
Tensor<P, 2> Tensor<P, 2>::matmulGpu(const Tensor<P, 2>& B) const {
    assert(cols == B.rows);

    toGPU();
    B.toGPU();

    using storageT = typename Precision<P>::storageT;

    Tensor<P, 2> C(rows, B.cols, storage -> device);

    const dim3 blockSize(MatmulConfig::BLOCK_N, MatmulConfig::BLOCK_M);
    const dim3 gridSize((B.cols + (MatmulConfig::BLOCK_N * 2) - 1) / (MatmulConfig::BLOCK_N), (rows + (MatmulConfig::BLOCK_M * 2) - 1) / (MatmulConfig::BLOCK_M));    //blockBim.x, blockDim.y

    matmul2x2Kernel<<<gridSize, blockSize>>>(gpuData(), B.gpuData(), C.gpuData(), static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(B.cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: matmulGpu: " + std::string(cudaGetErrorString(err)));
    }

    return C;
}

//Addition/Multiplication only on flattend j index -> for bias
template <typename P>
template <typename Functor, TensorAxis axis>
Tensor<P, 2> Tensor<P, 2>::broadcast(Tensor<P, 2>& B) const {
    toGPU();
    B.toGPU();

    using storageT = typename Precision<P>::storageT;
    
    Tensor<P, 2> C(rows, cols, storage -> device);

    const size_t N = rows * cols;
    constexpr int blockSize = 256;
    const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

    broadcastKernel<Functor, storageT, axis><<<numBlocks, blockSize>>>(gpuData(), B.gpuData(), C.gpuData(), static_cast<int>(rows), static_cast<int>(cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: broadcast: " + std::string(cudaGetErrorString(err)));
    }

    return C;
}

template <typename P>
template <typename Functor, TensorAxis axis>
Tensor<P, 2>& Tensor<P, 2>::broadcastInplace(const Tensor<P, 2>& B) {
    toGPU();
    B.toGPU();

    using storageT = typename Precision<P>::storageT;

    const size_t N = rows * cols;
    constexpr int blockSize = 256;
    const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

    broadcast_inplaceKernel<Functor, storageT, axis><<<numBlocks, blockSize>>>(gpuData(), B.gpuData(), static_cast<int>(rows), static_cast<int>(cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: broadcastInpalce: " + std::string(cudaGetErrorString(err)));
    }

    return *this;
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::addBroadcastRowGpu(Tensor<P, 2>& B) const {
    return this-> template broadcast<AddOp<typename Precision<P>::storageT>, TensorAxis::Row>(B);
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::addBroadcastColGpu(Tensor<P, 2>& B) const {
    return this-> template broadcast<AddOp<typename Precision<P>::storageT>, TensorAxis::Col>(B);
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::multiplyBroadcastRowGpu(Tensor<P, 2>& B) const {
    return this-> template broadcast<MulOp<typename Precision<P>::storageT>, TensorAxis::Row>(B);
}

template <typename P>
Tensor<P, 2>& Tensor<P, 2>::multiplyBroadcastRowInplaceGpu(const Tensor<P, 2>& B) {
    return this-> template broadcastInplace<MulOp<typename Precision<P>::storageT>, TensorAxis::Row>(B);
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::multiplyBroadcastColGpu(Tensor<P, 2>& B) const {
    return this-> template broadcast<MulOp<typename Precision<P>::storageT>, TensorAxis::Col>(B);
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::transposeGpu() const {
    toGPU();

    Tensor<P, 2> B(cols, rows, storage -> device);
    B.toGPU();

    const dim3 block(TransposeConfig::TILE_SIZE, TransposeConfig::TILE_SIZE);
    const dim3 grid((cols + TransposeConfig::TILE_SIZE - 1) / TransposeConfig::TILE_SIZE, (rows + TransposeConfig::TILE_SIZE - 1) / TransposeConfig::TILE_SIZE);

    transposeKernel<<<grid, block>>>(gpuData(), B.gpuData(), rows, cols);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: transoseGpu: " + std::string(cudaGetErrorString(err)));
    }

    return B;
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::sumRowsGpu() const {
    toGPU();

    Tensor<P, 2> result(1, cols, storage -> device);
    result.toGPU();

    using storageT = typename Precision<P>::storageT;

    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>(cols);
    const size_t shared = blockSize * sizeof(storageT);

    sumRowsKernel<<<gridSize, blockSize, shared>>>(gpuData(), result.gpuData(), static_cast<int>(rows), static_cast<int>(cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: sumRowsGpu: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::sumColsGpu() const {
    toGPU();

    Tensor<P, 2> result(rows, 1, storage -> device);
    result.toGPU();

    using storageT = typename Precision<P>::storageT;

    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>(rows);
    const size_t shared = blockSize * sizeof(storageT);

    sumColsKernel<<<gridSize, blockSize, shared>>>(gpuData(), result.gpuData(), static_cast<int>(cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: sumColsGpu" + std::string(cudaGetErrorString(err)));
    }

    return result;
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::minRowsGpu() const {
    toGPU();

    Tensor<P, 2> result(1, cols, storage -> device);
    result.toGPU();

    using storageT = typename Precision<P>::storageT;

    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>(cols);
    const size_t shared = blockSize * sizeof(storageT);

    minRowsKernel<<<gridSize, blockSize, shared>>>(gpuData(), result.gpuData(), static_cast<int>(rows), static_cast<int>(cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: minRowsGpu: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

template <typename P>
Tensor<P, 2> Tensor<P, 2>::maxRowsGpu() const {
    toGPU();

    Tensor<P, 2> result(1, cols, storage -> device);
    result.toGPU();

    using storageT = typename Precision<P>::storageT;
    
    int blockSize = 256;
    int gridSize = static_cast<int>(cols);
    size_t shared = blockSize * sizeof(storageT);

    maxRowsKernel<<<gridSize, blockSize, shared>>>(gpuData(), result.gpuData(), static_cast<int>(rows), static_cast<int>(cols));
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: maxRowsGpu" + std::string(cudaGetErrorString(err)));
    }

    return result;
}

template <typename P>
Tensor<size_t, 2> Tensor<P, 2>::argminRowsGpu() const {
    toGPU();

    Tensor<size_t, 2> result(1, cols, storage -> device);
    result.toGPU();

    using storageT = typename Precision<P>::storageT;
   
    constexpr int blockSize = 256;
    const int gridSize = cols;
    size_t shmem = blockSize * (sizeof(storageT) + sizeof(size_t));

    argminRowsKernel<<<gridSize, blockSize, shmem>>>(gpuData(), result.gpuData(), rows, cols);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: argminRowsGpu: " + std::string(cudaGetErrorString(err)));
    }

    return result;
}

template <typename P>
Tensor<size_t, 2> Tensor<P, 2>::argmaxRowsGpu() const {
    toGPU();

    Tensor<size_t, 2> result(1, cols, storage -> device);
    result.toGPU();

    using storageT = typename Precision<P>::storageT;

    constexpr int blockSize = 256;
    const int gridSize = cols;
    size_t shmem = blockSize * (sizeof(storageT) + sizeof(size_t));

    argmaxRowsKernel<<<gridSize, blockSize, shmem>>>(gpuData(), result.gpuData(), rows, cols);
    cudaError_t err = cudaDeviceSynchronize();

    if(err != cudaSuccess) {
        throw std::runtime_error("GPU: argmaxRowsGpu: " + std::string(cudaGetErrorString(err)));
    }
    
    return result;
}

//---------------- CUDA TEMPLATE INSTANTIATIONS ---------------------

template class Tensor<bf16, 2>;
template class Tensor<fp16, 2>;
template class Tensor<fp32, 2>;
template class Tensor<fp64, 2>;

}   //namespace alya