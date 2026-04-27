#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <cstddef>
#include <string>

#include <alya/core/tensor/Tensor.hpp>
#include <alya/core/tensor/TensorlinearOpsGpu.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/NumericLimits.cuh>
#include <alya/core/data/ArgMinMaxContainer.hpp>
#include <alya/core/ops/NumericalOps.cuh>

//----------- KERNEL ----------------

//elementwise
template <typename Op, typename T>
__global__ void elementwiseKernel(const T* A, const T* B, T* C, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        C[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], B[GLOBAL_IDX]);
    }
}

//elementwise Inplace
template <typename Op, typename T>
__global__ void elementwiseInplaceKernel(T* A, const T* B, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        A[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], B[GLOBAL_IDX]);
    }
}

//elementwise unary | ^2 & sqrt
template <typename Op, typename T>
__global__ void unaryKernel(const T* A, T* B, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        B[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX]);
    }
}

//elementwise unary Inplace | ^2 & sqrt
template <typename Op, typename T>
__global__ void unaryInplaceKernel(T* A, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        A[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX]);
    }
}

//Tensor-operation -> via scalar
template <typename Op, typename T>
__global__ void scalarKernel(const T* A, T* B, T scalar, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        B[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], scalar);
    }
}

//Tensor-operation Inplace -> via scalar
template <typename Op, typename T>
__global__ void scalarInplaceKernel(T* A, T scalar, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        A[GLOBAL_IDX] = Op::apply(A[GLOBAL_IDX], scalar);
    }
}

//L2 norm (frobenius norm)
template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void sumSquaredKernel(const T*__restrict__ data, T*__restrict__ partial, int N) {
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int GLOBAL_THREAD_START = GLOBAL_BLOCK_X * (BLOCK_SIZE * ITEMS_PER_THREAD) + LOCAL_THREAD_X;

    T sum = gpuZero<T>();

    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD * gridDim.x;

    for(int reduct_idx = GLOBAL_THREAD_START; reduct_idx < N; reduct_idx += stride) {
        #pragma unroll
        for(int k_idx = 0; k_idx < ITEMS_PER_THREAD; k_idx++) {
            int local_idx = reduct_idx + k_idx * BLOCK_SIZE;

            if(local_idx < N) {
                T v = data[local_idx];
                sum += v * v;
            }
        }
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ T warp_sums[BLOCK_SIZE / 32 + 1];

    const int lane = LOCAL_THREAD_X & 31;
    const int warp = LOCAL_THREAD_X >> 5;

    if(lane == 0) {
        warp_sums[warp] = sum;
    }

    __syncthreads();

    if(warp == 0) {
        sum = (LOCAL_THREAD_X < BLOCK_SIZE / 32) ? warp_sums[lane] : gpuZero<T>();

        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        if(LOCAL_THREAD_X == 0) {
            partial[GLOBAL_BLOCK_X] = sum;
        }
    }
}

//fils zero/ones
template <typename T>
__global__ void fillZeroKernel(T* A, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        A[GLOBAL_IDX] = gpuZero<T>();
    }
}

template <typename T>
__global__ void fillOneKernel(T* A, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        A[GLOBAL_IDX] = gpuOne<T>();
    }
}

//Squezzes Tensor into captivity
template <typename T>
__global__ void clipKernel(T* A, T minVal, T maxVal, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    if(GLOBAL_IDX < N) {
        A[GLOBAL_IDX] = fmin(fmax(A[GLOBAL_IDX], minVal), maxVal);
    }
}

template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void sumKernel(const T*__restrict__ data, T*__restrict__ partial, int N) {
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int GLOBAL_THREAD_START = GLOBAL_BLOCK_X * (BLOCK_SIZE * ITEMS_PER_THREAD) + LOCAL_THREAD_X;

    T sum = gpuZero<T>();

    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD * gridDim.x;

    for(int reduct_idx = GLOBAL_THREAD_START; reduct_idx < N; reduct_idx += stride) {
        #pragma unroll
        for(int k_idx = 0; k_idx < ITEMS_PER_THREAD; k_idx++) {
            int local_idx = reduct_idx + k_idx * BLOCK_SIZE;

            if(local_idx < N) {
                sum += data[local_idx];
            }
        }
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ T warp_sums[BLOCK_SIZE / 32 + 1];

    const int lane = LOCAL_THREAD_X & 31;
    const int warp = LOCAL_THREAD_X >> 5;

    if(lane == 0) {
        warp_sums[warp] = sum;
    }

    __syncthreads();

    if(warp == 0) {
        sum = (LOCAL_THREAD_X < BLOCK_SIZE / 32) ? warp_sums[lane] : gpuZero<T>();

        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        if(LOCAL_THREAD_X == 0) {
            partial[GLOBAL_BLOCK_X] = sum;
        }
    }
}

template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void minKernel(const T*__restrict__ data, T*__restrict__ partial, int N) {
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int GLOBAL_THREAD_START = GLOBAL_BLOCK_X * (BLOCK_SIZE * ITEMS_PER_THREAD) + LOCAL_THREAD_X;

    T sum = gpuMaxVal<T>();

    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD * gridDim.x;

    for(int reduct_idx = GLOBAL_THREAD_START; reduct_idx < N; reduct_idx += stride) {
        #pragma unroll
        for(int k_idx = 0; k_idx < ITEMS_PER_THREAD; k_idx++) {
            int local_idx = reduct_idx + k_idx * BLOCK_SIZE;

            if(local_idx < N) {
                sum = gpuMin(sum, data[local_idx]);
            }
        }
    }

    sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 16));
    sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 8));
    sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 4));
    sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 2));
    sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 1));

    __shared__ T warp_sums[BLOCK_SIZE / 32 + 1];

    const int lane = LOCAL_THREAD_X & 31;
    const int warp = LOCAL_THREAD_X >> 5;

    if(lane == 0) {
        warp_sums[warp] = sum;
    }

    __syncthreads();

    if(warp == 0) {
        sum = (LOCAL_THREAD_X < BLOCK_SIZE / 32) ? warp_sums[lane] : gpuZero<T>();

        sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 16));
        sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 8));
        sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 4));
        sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 2));
        sum = gpuMin(sum, __shfl_down_sync(0xffffffff, sum, 1));

        if(LOCAL_THREAD_X == 0) {
            partial[GLOBAL_BLOCK_X] = sum;
        }
    }
}

template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void maxKernel(const T*__restrict__ data, T*__restrict__ partial, int N) {
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int GLOBAL_THREAD_START = GLOBAL_BLOCK_X * (BLOCK_SIZE * ITEMS_PER_THREAD) + LOCAL_THREAD_X;

    T sum = gpuMinVal<T>();

    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD * gridDim.x;

    for(int reduct_idx = GLOBAL_THREAD_START; reduct_idx < N; reduct_idx += stride) {
        #pragma unroll
        for(int k_idx = 0; k_idx < ITEMS_PER_THREAD; k_idx++) {
            int local_idx = reduct_idx + k_idx * BLOCK_SIZE;

            if(local_idx < N) {
                sum = gpuMax(sum, data[local_idx]);
            }
        }
    }

    sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 16));
    sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 8));
    sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 4));
    sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 2));
    sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 1));

    __shared__ T warp_sums[BLOCK_SIZE / 32 + 1];

    const int lane = LOCAL_THREAD_X & 31;
    const int warp = LOCAL_THREAD_X >> 5;

    if(lane == 0) {
        warp_sums[warp] = sum;
    }

    __syncthreads();

    if(warp == 0) {
        sum = (LOCAL_THREAD_X < BLOCK_SIZE / 32) ? warp_sums[lane] : gpuZero<T>();

        sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 16));
        sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 8));
        sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 4));
        sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 2));
        sum = gpuMax(sum, __shfl_down_sync(0xffffffff, sum, 1));

        if(LOCAL_THREAD_X == 0) {
            partial[GLOBAL_BLOCK_X] = sum;
        }
    }
}

//converts tensorData to data + index -> for argMin/Max
template <typename T>
__global__ void TtoValIdx(const T* partial, ValIdx<T>* out, int N) {
    const int GLOBAL_IDX = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(GLOBAL_IDX < N) {
        out[GLOBAL_IDX].value = partial[GLOBAL_IDX];
        out[GLOBAL_IDX].idx = GLOBAL_IDX;
    }
}

template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void argminKernel(const ValIdx<T>*__restrict__ data, ValIdx<T>*__restrict__ partial, int N) {
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int GLOBAL_THREAD_START = GLOBAL_BLOCK_X * (BLOCK_SIZE * ITEMS_PER_THREAD) + LOCAL_THREAD_X;

    ValIdx<T> best;
    best.value = gpuMaxVal<T>();
    best.idx = -1;

    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD * gridDim.x;

    for(int reduct_idx = GLOBAL_THREAD_START; reduct_idx < N; reduct_idx += stride) {
        #pragma unroll
        for(int k_idx = 0; k_idx < ITEMS_PER_THREAD; k_idx++) {
            int local_idx = reduct_idx + k_idx * BLOCK_SIZE;

            if(local_idx < N) {
                ValIdx<T> v;
                v.value = data[local_idx].value;
                v.idx = local_idx;

                if(v.value < best.value || (v.value == best.value && v.idx < best.idx)) {
                    best = v;
                }
            }
        }
    }

    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        ValIdx<T> other;
        other.value = __shfl_down_sync(0xffffffff, best.value, offset);
        other.idx = __shfl_down_sync(0xffffffff, best.idx, offset);

        if(other.value < best.value || (other.value == best.value && other.idx < best.idx)) {
                best = other;
            }
    }

    __shared__ ValIdx<T> warp_vals[BLOCK_SIZE / 32];

    const int lane = LOCAL_THREAD_X & 31;
    const int warp = LOCAL_THREAD_X >> 5;

    if(lane == 0) {
        warp_vals[warp] = best;
    }

    __syncthreads();

    if(warp == 0) {
        if(LOCAL_THREAD_X < BLOCK_SIZE / 32) {
            best = warp_vals[lane];

        } else {
            best.value = gpuMaxVal<T>();
            best.idx = -1;
        }

        #pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1) {
            ValIdx<T> other;
            other.value = __shfl_down_sync(0xffffffff, best.value, offset);
            other.idx = __shfl_down_sync(0xffffffff, best.idx, offset);

            if(other.value < best.value || (other.value == best.value && other.idx < best.idx)) {
                best = other;
            }
    }

        if(LOCAL_THREAD_X == 0) {
            partial[GLOBAL_BLOCK_X] = best;
        }
    }
}

template <typename T, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void argmaxKernel(const ValIdx<T>*__restrict__ data, ValIdx<T>*__restrict__ partial, int N) {
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_X = blockIdx.x;
    const int GLOBAL_THREAD_START = GLOBAL_BLOCK_X * (BLOCK_SIZE * ITEMS_PER_THREAD) + LOCAL_THREAD_X;

    ValIdx<T> best;
    best.value = gpuMinVal<T>();
    best.idx = -1;

    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD * gridDim.x;

    for(int reduct_idx = GLOBAL_THREAD_START; reduct_idx < N; reduct_idx += stride) {
        #pragma unroll
        for(int k_idx = 0; k_idx < ITEMS_PER_THREAD; k_idx++) {
            int local_idx = reduct_idx + k_idx * BLOCK_SIZE;

            if(local_idx < N) {
                ValIdx<T> v;
                v.value = data[local_idx].value;
                v.idx = local_idx;

                if(v.value > best.value || (v.value == best.value && v.idx > best.idx)) {
                    best = v;
                }
            }
        }
    }

    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        ValIdx<T> other;
        other.value = __shfl_down_sync(0xffffffff, best.value, offset);
        other.idx = __shfl_down_sync(0xffffffff, best.idx, offset);

        if(other.value > best.value || (other.value == best.value && other.idx > best.idx)) {
                best = other;
            }
    }

    __shared__ ValIdx<T> warp_vals[BLOCK_SIZE / 32];

    const int lane = LOCAL_THREAD_X & 31;
    const int warp = LOCAL_THREAD_X >> 5;

    if(lane == 0) {
        warp_vals[warp] = best;
    }

    __syncthreads();

    if(warp == 0) {
        if(LOCAL_THREAD_X < BLOCK_SIZE / 32) {
            best = warp_vals[lane];

        } else {
            best.value = gpuMinVal<T>();
            best.idx = -1;
        }

        #pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1) {
            ValIdx<T> other;
            other.value = __shfl_down_sync(0xffffffff, best.value, offset);
            other.idx = __shfl_down_sync(0xffffffff, best.idx, offset);

            if(other.value > best.value || (other.value == best.value && other.idx > best.idx)) {
                best = other;
            }
    }

        if(LOCAL_THREAD_X == 0) {
            partial[GLOBAL_BLOCK_X] = best;
        }
    }
}
namespace alya::TensorLinearOpsGpu {
//------------- FUNCTIONS ---------------
    //elementwise
    template <TensorLike TensorType, typename Functor>
    TensorType elementwise(const TensorType& A, const TensorType& B) {
        A.toGPU();
        B.toGPU();

        TensorType C = A.emptyLike();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();
        const storageT* b = B.gpuData();
        storageT* c = C.gpuData();


        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        elementwiseKernel<Functor, storageT><<<numBlocks, blockSize>>>(a, b, c, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: elementwiseGpu: " + std::string(cudaGetErrorString(err)));
        }

        return C;
    }

    //elementwise Inplace
    template <TensorLike TensorType, typename Functor>
    TensorType& elementwiseInplace(TensorType& A, const TensorType& B) {
        A.toGPU();
        B.toGPU();

        using storageT = typename TensorType::storageT;

        storageT* a = A.gpuData();
        const storageT* b = B.gpuData();

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        elementwiseInplaceKernel<Functor, storageT><<<numBlocks, blockSize>>>(a, b, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: elementwiseInplaceGpu: " + std::string(cudaGetErrorString(err)));
        }

        return A;
    }

    //^2 & sqrt
    template <TensorLike TensorType, typename Functor>
    TensorType unary(const TensorType& A) {
        A.toGPU();

        TensorType B = A.emptyLike();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();
        storageT* b = B.gpuData();

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        unaryKernel<Functor, storageT><<<numBlocks, blockSize>>>(a, b, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: unaryGpu: " + std::string(cudaGetErrorString(err)));
        }

        return B;
    }

    //^2 & sqrt Inplace
    template <TensorLike TensorType, typename Functor>
    TensorType& unaryInplace(TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        storageT* a = A.gpuData();

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        unaryInplaceKernel<Functor, storageT><<<numBlocks, blockSize>>>(a, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: unaryInplaceGpu: " + std::string(cudaGetErrorString(err)));
        }

        return A;
    }

    //Tensor-operation -> via scalar
    template <TensorLike TensorType, typename Functor>
    TensorType scalar(const TensorType& A, const typename TensorType::computeT scalarV)  {
        A.toGPU();

        TensorType B = A.emptyLike();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();
        storageT* b = B.gpuData();

        storageT scalarVT = toStorage<storageT>(scalarV);

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        scalarKernel<Functor, storageT><<<numBlocks, blockSize>>>(a, b, scalarVT, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: scalarGpu: " + std::string(cudaGetErrorString(err)));
        }

        return B;
    }

    //Tensor-operation Inplace -> via scalar
    template <TensorLike TensorType, typename Functor>
    TensorType& scalarInplace(TensorType& A, const typename TensorType::computeT scalarV) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        storageT* a = A.gpuData();

        storageT scalarVT = toStorage<storageT>(scalarV);

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        scalarInplaceKernel<Functor, storageT><<<numBlocks, blockSize>>>(a, scalarVT, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: scalarInplaceGpu: " + std::string(cudaGetErrorString(err)));
        }

        return A;
    }

    //L2 norm (frobenius norm)
    template <TensorLike TensorType>
    TensorType::storageT normGpu(const TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();

        const int N = static_cast<int>(A.size());
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 8;
        int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
        
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

        const int maxBlocks = smCount * 4;
        gridSize = std::min(gridSize, maxBlocks);

        storageT* d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(storageT));
        
        sumSquaredKernel<storageT, blockSize, itemsPerThread><<<gridSize, blockSize>>>(a, d_partial, N);
        sumSquaredKernel<storageT, blockSize, itemsPerThread><<<1, blockSize>>>(d_partial, d_partial, gridSize);

        storageT result;
        cudaMemcpy(&result, d_partial, sizeof(storageT), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        return gpuSqrt<storageT>(result);
    }

    //divide l2 => final l2 normalization
    template <TensorLike TensorType>
    TensorType& normalizeGpu(TensorType& A) {
    using storageT = typename TensorType::storageT;
    storageT norm = normGpu<TensorType>(A);

        scalarInplace<TensorType, MulOp<storageT>>(A, gpuOne<storageT>() / norm);

        return A;
    }

    //fills zeros/ones
    template <TensorLike TensorType>
    TensorType& fillZeroGpu(TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        storageT* a = A.gpuData();
        
        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        fillZeroKernel<<<numBlocks, blockSize>>>(a, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: fillZeroGpu: " + std::string(cudaGetErrorString(err)));
        }

        return A;
    }

    template <TensorLike TensorType>
    TensorType& fillOneGpu(TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        storageT* a = A.gpuData();

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        fillOneKernel<<<numBlocks, blockSize>>>(a, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: fillOneGpu: " + std::string(cudaGetErrorString(err)));
        }

        return A;
    }

    template <TensorLike TensorType>
    TensorType& clipGpu(TensorType& A, typename TensorType::computeT minVal, typename TensorType::computeT maxVal) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        storageT* a = A.gpuData();

        storageT minValT = toStorage<storageT>(minVal);
        storageT maxValT = toStorage<storageT>(maxVal);

        const size_t N = A.size();
        constexpr int blockSize = 256;
        const int numBlocks = (static_cast<int>(N) + blockSize - 1) / blockSize;

        clipKernel<<<numBlocks, blockSize>>>(a, minValT, maxValT, static_cast<int>(N));
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            throw std::runtime_error("GPU: clipGpu: " + std::string(cudaGetErrorString(err)));
        }

        return A;
    }

    template <TensorLike TensorType>
    TensorType::storageT sumGpu(const TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();

        const int N = static_cast<int>(A.size());
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 8;
        int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
        
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

        const int maxBlocks = smCount * 4;
        gridSize = std::min(gridSize, maxBlocks);

        storageT* d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(storageT));
        
        sumKernel<storageT, blockSize, itemsPerThread><<<gridSize, blockSize>>>(a, d_partial, N);
        sumKernel<storageT, blockSize, itemsPerThread><<<1, blockSize>>>(d_partial, d_partial, gridSize);
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            cudaFree(d_partial);
            throw std::runtime_error("GPU: sum/meanGpu: " + std::string(cudaGetErrorString(err)));
        }

        storageT result;
        cudaMemcpy(&result, d_partial, sizeof(storageT), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        return result;
    }

    template <TensorLike TensorType>
    TensorType::storageT meanGpu(const TensorType& A) {
        using storageT = typename TensorType::storageT;

        return sumGpu<TensorType>(A) / static_cast<storageT>(A.size());
    }

    template <TensorLike TensorType>
    TensorType::storageT minGpu(const TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();

        const int N = static_cast<int>(A.size());
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 8;
        int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
        
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

        const int maxBlocks = smCount * 4;
        gridSize = std::min(gridSize, maxBlocks);

        storageT* d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(storageT));
        
        minKernel<storageT, blockSize, itemsPerThread><<<gridSize, blockSize>>>(a, d_partial, N);
        minKernel<storageT, blockSize, itemsPerThread><<<1, blockSize>>>(d_partial, d_partial, gridSize);
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            cudaFree(d_partial);
            throw std::runtime_error("GPU: minGpu: " + std::string(cudaGetErrorString(err)));
        }

        storageT result;
        cudaMemcpy(&result, d_partial, sizeof(storageT), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        return result;
    }

    template <TensorLike TensorType>
    TensorType::storageT maxGpu(const TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();

        const int N = static_cast<int>(A.size());
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 8;
        int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
        
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

        const int maxBlocks = smCount * 4;
        gridSize = std::min(gridSize, maxBlocks);

        storageT* d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(storageT));
        
        maxKernel<storageT, blockSize, itemsPerThread><<<gridSize, blockSize>>>(a, d_partial, N);
        maxKernel<storageT, blockSize, itemsPerThread><<<1, blockSize>>>(d_partial, d_partial, gridSize);
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            cudaFree(d_partial);
            throw std::runtime_error("GPU: maxGpu: " + std::string(cudaGetErrorString(err)));
        }

        storageT result;
        cudaMemcpy(&result, d_partial, sizeof(storageT), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        return result;
    }

    template <TensorLike TensorType>
    size_t argminGpu(const TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();

        const int N = static_cast<int>(A.size());
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 8;
        int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
        
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

        const int maxBlocks = smCount * 4;
        gridSize = std::min(gridSize, maxBlocks);

        ValIdx<storageT>* d_in;
        cudaMalloc(&d_in, N * sizeof(ValIdx<storageT>));

        ValIdx<storageT>* d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(ValIdx<storageT>));

        TtoValIdx<<<gridSize, blockSize>>>(a, d_in, N);
        
        argminKernel<storageT, blockSize, itemsPerThread><<<gridSize, blockSize>>>(d_in, d_partial, N);
        argminKernel<storageT, blockSize, itemsPerThread><<<1, blockSize>>>(d_partial, d_partial, gridSize);
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            cudaFree(d_partial);
            throw std::runtime_error("GPU: argminGpu: " + std::string(cudaGetErrorString(err)));
        }

        ValIdx<storageT> result;
        cudaMemcpy(&result, d_partial, sizeof(ValIdx<storageT>), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        return result.idx;
    }

    template <TensorLike TensorType>
    size_t argmaxGpu(const TensorType& A) {
        A.toGPU();

        using storageT = typename TensorType::storageT;

        const storageT* a = A.gpuData();

        const int N = static_cast<int>(A.size());
        constexpr int blockSize = 256;
        constexpr int itemsPerThread = 8;
        int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
        
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

        const int maxBlocks = smCount * 4;
        gridSize = std::min(gridSize, maxBlocks);

        ValIdx<storageT>* d_in;
        cudaMalloc(&d_in, N * sizeof(ValIdx<storageT>));

        ValIdx<storageT>* d_partial;
        cudaMalloc(&d_partial, gridSize * sizeof(ValIdx<storageT>));

        TtoValIdx<<<gridSize, blockSize>>>(a, d_in, N);
        
        argmaxKernel<storageT, blockSize, itemsPerThread><<<gridSize, blockSize>>>(d_in, d_partial, N);
        argmaxKernel<storageT, blockSize, itemsPerThread><<<1, blockSize>>>(d_partial, d_partial, gridSize);
        cudaError_t err = cudaDeviceSynchronize();

        if(err != cudaSuccess) {
            cudaFree(d_partial);
            throw std::runtime_error("GPU: argmaxGpu: " + std::string(cudaGetErrorString(err)));
        }

        ValIdx<storageT> result;
        cudaMemcpy(&result, d_partial, sizeof(ValIdx<storageT>), cudaMemcpyDeviceToHost);
        cudaFree(d_partial);

        return result.idx;
    }
}   //namespace alya::TensorLinearOpsGpu
//------------------------------------- CUDA TEMPLATE INSTANTIATIONS ---------------------------------------------------

namespace alya::TensorLinearOpsGpu {

template Tensor<bf16, 2> TensorLinearOpsGpu::elementwise<Tensor<bf16, 2>, MulOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::elementwise<Tensor<fp16, 2>, MulOp<__half>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::elementwise<Tensor<fp32, 2>, MulOp<float>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::elementwise<Tensor<fp64, 2>, MulOp<double>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::elementwise<Tensor<bf16, 2>, DivOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::elementwise<Tensor<fp16, 2>, DivOp<__half>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::elementwise<Tensor<fp32, 2>, DivOp<float>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::elementwise<Tensor<fp64, 2>, DivOp<double>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::elementwise<Tensor<bf16, 2>, AddOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::elementwise<Tensor<fp16, 2>, AddOp<__half>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::elementwise<Tensor<fp32, 2>, AddOp<float>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::elementwise<Tensor<fp64, 2>, AddOp<double>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::elementwise<Tensor<bf16, 2>, SubOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::elementwise<Tensor<fp16, 2>, SubOp<__half>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::elementwise<Tensor<fp32, 2>, SubOp<float>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::elementwise<Tensor<fp64, 2>, SubOp<double>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);


template Tensor<bf16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<bf16, 2>, MulOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp16, 2>, MulOp<__half>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp32, 2>, MulOp<float>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp64, 2>, MulOp<double>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<bf16, 2>, DivOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp16, 2>, DivOp<__half>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp32, 2>, DivOp<float>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp64, 2>, DivOp<double>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<bf16, 2>, AddOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp16, 2>, AddOp<__half>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp32, 2>, AddOp<float>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp64, 2>, AddOp<double>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<bf16, 2>, SubOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp16, 2>, SubOp<__half>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp32, 2>, SubOp<float>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::elementwiseInplace<Tensor<fp64, 2>, SubOp<double>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);


template Tensor<bf16, 2> TensorLinearOpsGpu::unary<Tensor<bf16, 2>, SquareOp<__nv_bfloat16>>(const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::unary<Tensor<fp16, 2>, SquareOp<__half>>(const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::unary<Tensor<fp32, 2>, SquareOp<float>>(const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::unary<Tensor<fp64, 2>, SquareOp<double>>(const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::unary<Tensor<bf16, 2>, SqrtOp<__nv_bfloat16>>(const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::unary<Tensor<fp16, 2>, SqrtOp<__half>>(const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::unary<Tensor<fp32, 2>, SqrtOp<float>>(const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::unary<Tensor<fp64, 2>, SqrtOp<double>>(const Tensor<fp64, 2>&);


template Tensor<bf16, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<bf16, 2>, SquareOp<__nv_bfloat16>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<fp16, 2>, SquareOp<__half>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<fp32, 2>, SquareOp<float>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<fp64, 2>, SquareOp<double>>(Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<bf16, 2>, SqrtOp<__nv_bfloat16>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<fp16, 2>, SqrtOp<__half>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<fp32, 2>, SqrtOp<float>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::unaryInplace<Tensor<fp64, 2>, SqrtOp<double>>(Tensor<fp64, 2>&);


template Tensor<bf16, 2> TensorLinearOpsGpu::scalar<Tensor<bf16, 2>, MulOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalar<Tensor<fp16, 2>, MulOp<__half>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalar<Tensor<fp32, 2>, MulOp<float>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalar<Tensor<fp64, 2>, MulOp<double>>(const Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2> TensorLinearOpsGpu::scalar<Tensor<bf16, 2>, DivOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalar<Tensor<fp16, 2>, DivOp<__half>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalar<Tensor<fp32, 2>, DivOp<float>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalar<Tensor<fp64, 2>, DivOp<double>>(const Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2> TensorLinearOpsGpu::scalar<Tensor<bf16, 2>, AddOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalar<Tensor<fp16, 2>, AddOp<__half>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalar<Tensor<fp32, 2>, AddOp<float>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalar<Tensor<fp64, 2>, AddOp<double>>(const Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2> TensorLinearOpsGpu::scalar<Tensor<bf16, 2>, SubOp<__nv_bfloat16>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalar<Tensor<fp16, 2>, SubOp<__half>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalar<Tensor<fp32, 2>, SubOp<float>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalar<Tensor<fp64, 2>, SubOp<double>>(const Tensor<fp64, 2>&, const double scalarV);


template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<bf16, 2>, MulOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp16, 2>, MulOp<__half>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp32, 2>, MulOp<float>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp64, 2>, MulOp<double>>(Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<bf16, 2>, DivOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp16, 2>, DivOp<__half>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp32, 2>, DivOp<float>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp64, 2>, DivOp<double>>(Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<bf16, 2>, AddOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp16, 2>, AddOp<__half>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp32, 2>, AddOp<float>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp64, 2>, AddOp<double>>(Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<bf16, 2>, SubOp<__nv_bfloat16>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp16, 2>, SubOp<__half>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp32, 2>, SubOp<float>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarInplace<Tensor<fp64, 2>, SubOp<double>>(Tensor<fp64, 2>&, const double scalarV);



template Tensor<bf16, 2> TensorLinearOpsGpu::hadamardGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::hadamardGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::hadamardGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::hadamardGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::divideGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::divideGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::divideGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::divideGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::addGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::addGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::addGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::addGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::subtractGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::subtractGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::subtractGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::subtractGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const Tensor<fp64, 2>&);


template Tensor<bf16, 2>& TensorLinearOpsGpu::hadamardInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::hadamardInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::hadamardInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::hadamardInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::divideInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::divideInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::divideInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::divideInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::addInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::addInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::addInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::addInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::subtractInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::subtractInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::subtractInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::subtractInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const Tensor<fp64, 2>&);


template Tensor<bf16, 2> TensorLinearOpsGpu::squareGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::squareGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::squareGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::squareGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

template Tensor<bf16, 2> TensorLinearOpsGpu::sqrtGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template Tensor<fp16, 2> TensorLinearOpsGpu::sqrtGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template Tensor<fp32, 2> TensorLinearOpsGpu::sqrtGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template Tensor<fp64, 2> TensorLinearOpsGpu::sqrtGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);


template Tensor<bf16, 2>& TensorLinearOpsGpu::squareInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::squareInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::squareInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::squareInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::sqrtInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::sqrtInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::sqrtInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::sqrtInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&);



template Tensor<bf16, 2> TensorLinearOpsGpu::scalarScaleGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalarScaleGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalarScaleGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalarScaleGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2> TensorLinearOpsGpu::scalarDivideGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalarDivideGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalarDivideGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalarDivideGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2> TensorLinearOpsGpu::scalarAddGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalarAddGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalarAddGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalarAddGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2> TensorLinearOpsGpu::scalarSubtractGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2> TensorLinearOpsGpu::scalarSubtractGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2> TensorLinearOpsGpu::scalarSubtractGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2> TensorLinearOpsGpu::scalarSubtractGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&, const double scalarV);


template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarScaleInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarScaleInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarScaleInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarScaleInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarDivideInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarDivideInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarDivideInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarDivideInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarAddInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarAddInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarAddInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarAddInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const double scalarV);

template Tensor<bf16, 2>& TensorLinearOpsGpu::scalarSubtractInplaceGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, const float scalarV);
template Tensor<fp16, 2>& TensorLinearOpsGpu::scalarSubtractInplaceGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, const float scalarV);
template Tensor<fp32, 2>& TensorLinearOpsGpu::scalarSubtractInplaceGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, const float scalarV);
template Tensor<fp64, 2>& TensorLinearOpsGpu::scalarSubtractInplaceGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, const double scalarV);


template __nv_bfloat16 TensorLinearOpsGpu::normGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template __half TensorLinearOpsGpu::normGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template float TensorLinearOpsGpu::normGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template double TensorLinearOpsGpu::normGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::normalizeGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::normalizeGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::normalizeGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::normalizeGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&);


template Tensor<bf16, 2>& TensorLinearOpsGpu::fillZeroGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::fillZeroGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::fillZeroGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::fillZeroGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::fillOneGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&);
template Tensor<fp16, 2>& TensorLinearOpsGpu::fillOneGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&);
template Tensor<fp32, 2>& TensorLinearOpsGpu::fillOneGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&);
template Tensor<fp64, 2>& TensorLinearOpsGpu::fillOneGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&);

template Tensor<bf16, 2>& TensorLinearOpsGpu::clipGpu<Tensor<bf16, 2>>(Tensor<bf16, 2>&, float minVal, float maxVal);
template Tensor<fp16, 2>& TensorLinearOpsGpu::clipGpu<Tensor<fp16, 2>>(Tensor<fp16, 2>&, float minVal, float maxVal);
template Tensor<fp32, 2>& TensorLinearOpsGpu::clipGpu<Tensor<fp32, 2>>(Tensor<fp32, 2>&, float minVal, float maxVal);
template Tensor<fp64, 2>& TensorLinearOpsGpu::clipGpu<Tensor<fp64, 2>>(Tensor<fp64, 2>&, double minVal, double maxVal);


template __nv_bfloat16 TensorLinearOpsGpu::sumGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template __half TensorLinearOpsGpu::sumGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template float TensorLinearOpsGpu::sumGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template double TensorLinearOpsGpu::sumGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

template __nv_bfloat16 TensorLinearOpsGpu::meanGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template __half TensorLinearOpsGpu::meanGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template float TensorLinearOpsGpu::meanGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template double TensorLinearOpsGpu::meanGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);


template __nv_bfloat16 TensorLinearOpsGpu::minGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template __half TensorLinearOpsGpu::minGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template float TensorLinearOpsGpu::minGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template double TensorLinearOpsGpu::minGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

template __nv_bfloat16 TensorLinearOpsGpu::maxGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template __half TensorLinearOpsGpu::maxGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template float TensorLinearOpsGpu::maxGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template double TensorLinearOpsGpu::maxGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

template size_t TensorLinearOpsGpu::argminGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template size_t TensorLinearOpsGpu::argminGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template size_t TensorLinearOpsGpu::argminGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template size_t TensorLinearOpsGpu::argminGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

template size_t TensorLinearOpsGpu::argmaxGpu<Tensor<bf16, 2>>(const Tensor<bf16, 2>&);
template size_t TensorLinearOpsGpu::argmaxGpu<Tensor<fp16, 2>>(const Tensor<fp16, 2>&);
template size_t TensorLinearOpsGpu::argmaxGpu<Tensor<fp32, 2>>(const Tensor<fp32, 2>&);
template size_t TensorLinearOpsGpu::argmaxGpu<Tensor<fp64, 2>>(const Tensor<fp64, 2>&);

}   //namespace alya::TensorLinearOpsGpu