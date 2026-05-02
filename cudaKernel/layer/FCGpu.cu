#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <string>

#include <alya/layer/FC.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/activation/ActivationOps.cuh>
#include <alya/profiling/CudaCheck.cuh>

template <typename Op, typename T>
__device__ inline T applyActivation(T x) { return Op::apply(x); }

namespace MatmulConfig {
    constexpr int BLOCK_M = 8;
    constexpr int BLOCK_N = 8;
    constexpr int MATMUL_VALUES_K = 8;
    constexpr int THREAD_TILE_SIZE = 2;
};

//---------- KERNEL --------- 
template <typename T, typename ActOp>
__global__ void FusedForwardKernel(const T* __restrict__ input, const T* __restrict__ weights, const T* __restrict__ bias, T* __restrict__ z, T* __restrict__ a, int M, int K, int N) {
    const int THREAD_VALUES_Y = MatmulConfig::BLOCK_M * MatmulConfig::THREAD_TILE_SIZE;
    const int THREAD_VALUES_X = MatmulConfig::BLOCK_N * MatmulConfig::THREAD_TILE_SIZE;
    const int LOCAL_THREAD_Y = threadIdx.y;
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_Y = blockIdx.y * THREAD_VALUES_Y + LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE;
    const int GLOBAL_BLOCK_X = blockIdx.x * THREAD_VALUES_X + LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE;

    __shared__ T As[THREAD_VALUES_Y][MatmulConfig::MATMUL_VALUES_K + 1];
    __shared__ T Bs[THREAD_VALUES_X][MatmulConfig::MATMUL_VALUES_K + 1];

    T v00 = gpuZero<T>(), v01 = gpuZero<T>();
    T v10 = gpuZero<T>(), v11 = gpuZero<T>();
    T b0 = gpuZero<T>(), b1 = gpuZero<T>();

    const int K_TILES = (K + MatmulConfig::MATMUL_VALUES_K - 1) / MatmulConfig::MATMUL_VALUES_K;

    for(size_t k_tile = 0; k_tile < K_TILES; k_tile++) {
        const int K_TILE_IDX = k_tile * MatmulConfig::MATMUL_VALUES_K;

        const int A_Y0 = GLOBAL_BLOCK_Y;
        const int A_Y1 = GLOBAL_BLOCK_Y + 1;
        const int A_X = LOCAL_THREAD_X + K_TILE_IDX;

        const int B_X0 = GLOBAL_BLOCK_X;
        const int B_X1 = GLOBAL_BLOCK_X + 1;
        const int B_Y = LOCAL_THREAD_Y + K_TILE_IDX;

        As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_X] = (A_Y0 < M && A_X < K) ? input[A_Y0 * K + A_X] : gpuZero<T>();
        As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_X] = (A_Y1 < M && A_X < K) ? input[A_Y1 * K + A_X] : gpuZero<T>();

        Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_Y] = (B_X0 < N && B_Y < K) ? weights[B_Y * N + B_X0] : gpuZero<T>();
        Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_Y] = (B_X1 < N && B_Y < K) ? weights[B_Y * N + B_X1] : gpuZero<T>();

        __syncthreads();

        #pragma unroll
        for(size_t k = 0; k < MatmulConfig::MATMUL_VALUES_K; k++) {
            T temp_a0 = As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE][k];
            T temp_a1 = As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE + 1][k];
            T temp_b0 = Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE][k];
            T temp_b1 = Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE + 1][k];

            v00 += temp_a0 * temp_b0;
            v01 += temp_a0 * temp_b1;
            v10 += temp_a1 * temp_b0;
            v11 += temp_a1 * temp_b1;
        }

        __syncthreads();
    }

    b0 = (GLOBAL_BLOCK_X < N) ? bias[GLOBAL_BLOCK_X] : gpuZero<T>();
    b1 = (GLOBAL_BLOCK_X + 1 < N) ? bias[GLOBAL_BLOCK_X + 1] : gpuZero<T>();

    v00 += b0;
    v01 += b1;
    v10 += b0;
    v11 += b1;

    T z00 = v00, z01 = v01;
    T z10 = v10, z11 = v11;

    v00 = applyActivation<ActOp>(z00);
    v01 = applyActivation<ActOp>(z01);
    v10 = applyActivation<ActOp>(z10);
    v11 = applyActivation<ActOp>(z11);

    if(GLOBAL_BLOCK_Y < M && GLOBAL_BLOCK_X < N) {
        z[GLOBAL_BLOCK_Y * N + GLOBAL_BLOCK_X] = z00;
        a[GLOBAL_BLOCK_Y * N + GLOBAL_BLOCK_X] = v00;
    }

    if(GLOBAL_BLOCK_Y < M && GLOBAL_BLOCK_X + 1 < N) {
        z[GLOBAL_BLOCK_Y * N + GLOBAL_BLOCK_X + 1] = z01;
        a[GLOBAL_BLOCK_Y * N + GLOBAL_BLOCK_X + 1] = v01;
    }

    if((GLOBAL_BLOCK_Y + 1) < M && GLOBAL_BLOCK_X < N) {
        z[(GLOBAL_BLOCK_Y + 1) * N + GLOBAL_BLOCK_X] = z10;
        a[(GLOBAL_BLOCK_Y + 1) * N + GLOBAL_BLOCK_X] = v10;
    }

    if((GLOBAL_BLOCK_Y + 1) < M && (GLOBAL_BLOCK_X + 1) < N) {
        z[(GLOBAL_BLOCK_Y + 1) * N + GLOBAL_BLOCK_X + 1] = z11;
        a[(GLOBAL_BLOCK_Y + 1) * N + GLOBAL_BLOCK_X + 1] = v11;
    }
}

template <typename T, typename ActOp>
__global__ void FusedDeltaDbKernel(const T* __restrict__ gradOut, T* __restrict__ z, const T* __restrict__ a, T* __restrict__ db, int rows, int cols) {
    extern __shared__ unsigned char cache[];
    T* shmem = reinterpret_cast<T*>(cache);

    const int GLOBAL_BLOCK_X = blockIdx.x;
    if(GLOBAL_BLOCK_X >= cols) { return; }

    const int LOCAL_THREAD_X = threadIdx.x;
    T local_thread_sum = gpuZero<T>();

    for(int thread_x = LOCAL_THREAD_X; thread_x < rows; thread_x += blockDim.x) {
        int idx = thread_x * cols + GLOBAL_BLOCK_X;
        T value = ActOp::backwardScalar(gradOut[idx], z[idx], a[idx]);

        z[idx] = value;

        local_thread_sum += value;
    }

    shmem[LOCAL_THREAD_X] = local_thread_sum;

    __syncthreads();

    for(int reductIdx = blockDim.x / 2; reductIdx >= 32; reductIdx >>= 1) {
        if(LOCAL_THREAD_X < reductIdx) {
            shmem[LOCAL_THREAD_X] += shmem[LOCAL_THREAD_X + reductIdx];
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
        db[GLOBAL_BLOCK_X] = val;
    }
}

template <typename T> 
__global__ void dwKernel(const T* __restrict__ input, const T* __restrict__ z, T* __restrict__ dw, int M, int K, int N) {
    const int THREAD_VALUES_Y = MatmulConfig::BLOCK_M * 2;
    const int THREAD_VALUES_X = MatmulConfig::BLOCK_N * 2;
    const int LOCAL_THREAD_Y = threadIdx.y;
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_K = blockIdx.y * THREAD_VALUES_Y + LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE;
    const int GLOBAL_BLOCK_N = blockIdx.x * THREAD_VALUES_X + LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE;

    __shared__ T As[THREAD_VALUES_Y][MatmulConfig::MATMUL_VALUES_K + 1];
    __shared__ T Bs[THREAD_VALUES_X][MatmulConfig::MATMUL_VALUES_K + 1];

    T v00 = gpuZero<T>(), v01 = gpuZero<T>();
    T v10 = gpuZero<T>(), v11 = gpuZero<T>();

    const int M_TILES = (M + MatmulConfig::MATMUL_VALUES_K - 1) / MatmulConfig::MATMUL_VALUES_K;

    for(size_t m_tile = 0; m_tile < M_TILES; m_tile++) {
        const int M_TILE_IDX = m_tile * MatmulConfig::MATMUL_VALUES_K;

        const int A_K0 = GLOBAL_BLOCK_K;
        const int A_K1 = GLOBAL_BLOCK_K + 1;
        const int A_M = LOCAL_THREAD_X + M_TILE_IDX;

        const int B_N0 = GLOBAL_BLOCK_N;
        const int B_N1 = GLOBAL_BLOCK_N + 1;
        const int B_M = LOCAL_THREAD_Y + M_TILE_IDX;

        As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_X] = (A_K0 < K && A_M < M) ? input[A_M * K + A_K0] : gpuZero<T>();
        As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_X] = (A_K1 < K && A_M < M) ? input[A_M * K + A_K1] : gpuZero<T>();

        Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_Y] = (B_M < M && B_N0 < N) ? z[B_M * N + B_N0] : gpuZero<T>();
        Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_Y] = (B_M < M && B_N1 < N) ? z[B_M * N + B_N1] : gpuZero<T>();

        __syncthreads();

        #pragma unroll
        for(size_t m = 0; m < MatmulConfig::MATMUL_VALUES_K; m++) {
            T temp_a0 = As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE][m];
            T temp_a1 = As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE + 1][m];
            T temp_b0 = Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE][m];
            T temp_b1 = Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE + 1][m];

            v00 += temp_a0 * temp_b0;
            v01 += temp_a0 * temp_b1;
            v10 += temp_a1 * temp_b0;
            v11 += temp_a1 * temp_b1;
        }

        __syncthreads();
    }

    if(GLOBAL_BLOCK_K < K && GLOBAL_BLOCK_N < N) {
        dw[GLOBAL_BLOCK_K * N + GLOBAL_BLOCK_N] = v00;
    }

    if(GLOBAL_BLOCK_K < K && GLOBAL_BLOCK_N + 1 < N) {
        dw[GLOBAL_BLOCK_K * N + GLOBAL_BLOCK_N + 1] = v01;
    }

    if((GLOBAL_BLOCK_K + 1) < K && GLOBAL_BLOCK_N < N) {
        dw[(GLOBAL_BLOCK_K + 1) * N + GLOBAL_BLOCK_N] = v10;
    }

    if((GLOBAL_BLOCK_K + 1) < K && (GLOBAL_BLOCK_N + 1) < N) {
        dw[(GLOBAL_BLOCK_K + 1) * N + GLOBAL_BLOCK_N + 1] = v11;
    }
}

template <typename T>   //z = [M, N], weights = [K, N], grad_in = [M, K] -> grad_in = z * weights^T
__global__ void gradInKernel(const T* __restrict__ z, const T* __restrict__ weights, T* __restrict__ gradIn, int M, int K, int N) {
    const int THREAD_VALUES_M = MatmulConfig::BLOCK_M * MatmulConfig::THREAD_TILE_SIZE;
    const int THREAD_VALUES_K = MatmulConfig::BLOCK_N * MatmulConfig::THREAD_TILE_SIZE;
    const int LOCAL_THREAD_Y = threadIdx.y;
    const int LOCAL_THREAD_X = threadIdx.x;
    const int GLOBAL_BLOCK_M = blockIdx.y * THREAD_VALUES_M + LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE;
    const int GLOBAL_BLOCK_K = blockIdx.x * THREAD_VALUES_K + LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE;

    __shared__ T As[THREAD_VALUES_M][MatmulConfig::MATMUL_VALUES_K + 1];
    __shared__ T Bs[THREAD_VALUES_K][MatmulConfig::MATMUL_VALUES_K + 1];

    T v00 = gpuZero<T>(), v01 = gpuZero<T>();
    T v10 = gpuZero<T>(), v11 = gpuZero<T>();

    const int N_TILES = (N + MatmulConfig::MATMUL_VALUES_K - 1) / MatmulConfig::MATMUL_VALUES_K;

    for(size_t n_tile = 0; n_tile < N_TILES; n_tile++) {
        int N_IDX_BASE = n_tile * MatmulConfig::MATMUL_VALUES_K;
        int N_IDX_A = N_IDX_BASE + LOCAL_THREAD_X;
        int N_IDX_B = N_IDX_BASE + LOCAL_THREAD_Y;

        const int A_M0 = GLOBAL_BLOCK_M;
        const int A_M1 = GLOBAL_BLOCK_M + 1;

        const int B_K0 = GLOBAL_BLOCK_K;
        const int B_K1 = GLOBAL_BLOCK_K + 1;

        As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_X] = (A_M0 < M && N_IDX_A < N) ? z[A_M0 * N + N_IDX_A] : gpuZero<T>();
        As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_X] = (A_M1 < M && N_IDX_A < N) ? z[A_M1 * N + N_IDX_A] : gpuZero<T>();

        Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE][LOCAL_THREAD_Y] = (B_K0 < K && N_IDX_B < N) ? weights[B_K0 * N + N_IDX_B] : gpuZero<T>();
        Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE + 1][LOCAL_THREAD_Y] = (B_K1 < K && N_IDX_B < N) ? weights[B_K1 * N + N_IDX_B] : gpuZero<T>();
        __syncthreads();

        #pragma unroll
        for(size_t n = 0; n < MatmulConfig::MATMUL_VALUES_K; n++) {
            T temp_a0 = As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE][n];
            T temp_a1 = As[LOCAL_THREAD_Y * MatmulConfig::THREAD_TILE_SIZE + 1][n];
            T temp_b0 = Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE][n];
            T temp_b1 = Bs[LOCAL_THREAD_X * MatmulConfig::THREAD_TILE_SIZE + 1][n];

            v00 += temp_a0 * temp_b0;
            v01 += temp_a0 * temp_b1;
            v10 += temp_a1 * temp_b0;
            v11 += temp_a1 * temp_b1;
        }

        __syncthreads();
    }

    if(GLOBAL_BLOCK_M < M && GLOBAL_BLOCK_K < K) {
        gradIn[GLOBAL_BLOCK_M * K + GLOBAL_BLOCK_K] = v00;
    }

    if(GLOBAL_BLOCK_M < M && GLOBAL_BLOCK_K + 1 < K) {
        gradIn[GLOBAL_BLOCK_M * K + GLOBAL_BLOCK_K + 1] = v01;
    }

    if((GLOBAL_BLOCK_M + 1) < M && GLOBAL_BLOCK_K < K) {
        gradIn[(GLOBAL_BLOCK_M + 1) * K + GLOBAL_BLOCK_K] = v10;
    }

    if((GLOBAL_BLOCK_M + 1) < M && (GLOBAL_BLOCK_K + 1) < K) {
        gradIn[(GLOBAL_BLOCK_M + 1) * K + GLOBAL_BLOCK_K + 1] = v11;
    }
}

namespace alya {

//----------- FUNKTIONS -----------
template <typename P, template <typename> class ActOp>
Tensor<P, 2> FC<P, ActOp>::forwardGpu(const Tensor<P, 2>& inputIn) {
    if(inputIn.device().type == DeviceType::GPU) {
        cache.input = inputIn;
    } else {
        cache.input = inputIn.clone();
        cache.input.toGPU();
    }

    if(cache.act.z.numRows() != cache.input.numRows() || cache.act.z.numCols() != weights.numCols() || cache.act.z.device().type != weights.device().type) {
        cache.act.z = Tensor<P, 2>(cache.input.numRows(), weights.numCols(), weights.device());
    }
    if(cache.act.a.numRows() != cache.input.numRows() || cache.act.a.numCols() != weights.numCols() || cache.act.a.device().type != weights.device().type) {
        cache.act.a = Tensor<P, 2>(cache.input.numRows(), weights.numCols(), weights.device());
    }

    using ActOpT = ActOp<storageT>;

    const storageT* d_input = cache.input.gpuData();
    const storageT* d_weights = weights.gpuData();
    const storageT* d_bias = bias.gpuData();
    storageT* d_z = cache.act.z.gpuData();
    storageT* d_a = cache.act.a.gpuData();

    dim3 blockSize(MatmulConfig::BLOCK_N, MatmulConfig::BLOCK_M);
    dim3 numBlocks(
        (weights.numCols() + (MatmulConfig::BLOCK_N * 2) - 1) / (MatmulConfig::BLOCK_N * 2),
        (cache.input.numRows() + (MatmulConfig::BLOCK_M * 2) - 1) / (MatmulConfig::BLOCK_M * 2));

    FusedForwardKernel<storageT, ActOpT><<<numBlocks, blockSize>>>(
        d_input,
        d_weights,
        d_bias,
        d_z,
        d_a,
        static_cast<int>(cache.input.numRows()),
        static_cast<int>(cache.input.numCols()),
        static_cast<int>(weights.numCols()));

    CUDA_CHECK(cudaDeviceSynchronize());

    return cache.act.a;
}

template <typename P, template <typename> class ActOp>
Tensor<P, 2> FC<P, ActOp>::backwardGpu(const Tensor<P, 2>& gradOut) {
    Tensor<P, 2> gradOutGpu = (gradOut.device().type == DeviceType::GPU) ? gradOut : gradOut.clone();
    if(gradOutGpu.device().type != DeviceType::GPU) {
        gradOutGpu.toGPU();
    }

    using storageT = typename Precision<P>::storageT;
    using ActOpT = ActOp<storageT>;

    if(dw.numRows() != cache.input.numCols() || dw.numCols() != cache.act.z.numCols() || dw.device().type != cache.act.z.device().type) {
        dw = Tensor<P, 2>(cache.input.numCols(), cache.act.z.numCols(), cache.act.z.device());
    }
    if(db.numRows() != 1 || db.numCols() != cache.act.z.numCols() || db.device().type != cache.act.z.device().type) {
        db = Tensor<P, 2>(1, cache.act.z.numCols(), cache.act.z.device());
    }

    Tensor<P, 2> gradInput(cache.input.numRows(), cache.input.numCols(), cache.input.device());

    const storageT* d_gradOut = gradOutGpu.gpuData();
    storageT* d_z = cache.act.z.gpuData();
    const storageT* d_a = cache.act.a.gpuData();
    storageT* d_db = db.gpuData();
    const storageT* d_input = cache.input.gpuData();
    const storageT* d_weights = weights.gpuData();
    storageT* d_dw = dw.gpuData();
    storageT* d_gradInput = gradInput.gpuData();

    int blockSizeDb = 256;
    int gridSizeDb = static_cast<int>(cache.act.a.numCols());
    int shmem_bytes = blockSizeDb * sizeof(storageT);
    dim3 blockSize(MatmulConfig::BLOCK_N, MatmulConfig::BLOCK_M);
    dim3 gridSizeDw(
        (cache.act.z.numCols() + (MatmulConfig::BLOCK_N * 2) - 1) / (MatmulConfig::BLOCK_N * 2),
        (cache.input.numCols() + (MatmulConfig::BLOCK_M * 2) - 1) / (MatmulConfig::BLOCK_M * 2));
    dim3 gridSizeGradInput(
        (weights.numRows() + (MatmulConfig::BLOCK_N * 2) - 1) / (MatmulConfig::BLOCK_N * 2),
        (cache.act.z.numRows() + (MatmulConfig::BLOCK_M * 2) - 1) / (MatmulConfig::BLOCK_M * 2));

    FusedDeltaDbKernel<storageT, ActOpT><<<gridSizeDb, blockSizeDb, shmem_bytes>>>(
        d_gradOut,
        d_z,
        d_a,
        d_db,
        static_cast<int>(cache.act.a.numRows()),
        static_cast<int>(cache.act.a.numCols()));
    dwKernel<storageT><<<gridSizeDw, blockSize>>>(
        d_input,
        d_z,
        d_dw,
        static_cast<int>(cache.input.numRows()),
        static_cast<int>(cache.input.numCols()),
        static_cast<int>(cache.act.z.numCols()));
    gradInKernel<storageT><<<gridSizeGradInput, blockSize>>>(
        d_z,
        d_weights,
        d_gradInput,
        static_cast<int>(cache.act.z.numRows()),
        static_cast<int>(weights.numRows()),
        static_cast<int>(cache.act.z.numCols()));

    CUDA_CHECK(cudaDeviceSynchronize());

    return gradInput;
}

template class FC<bf16, LinearOp>;
template class FC<fp16, LinearOp>;
template class FC<fp32, LinearOp>;
template class FC<fp64, LinearOp>;

template class FC<bf16, ReLuOp>;
template class FC<fp16, ReLuOp>;
template class FC<fp32, ReLuOp>;
template class FC<fp64, ReLuOp>;

template class FC<bf16, LeakyReLuOp>;
template class FC<fp16, LeakyReLuOp>;
template class FC<fp32, LeakyReLuOp>;
template class FC<fp64, LeakyReLuOp>;

template class FC<bf16, ELUOp>;
template class FC<fp16, ELUOp>;
template class FC<fp32, ELUOp>;
template class FC<fp64, ELUOp>;

template class FC<bf16, SigmoidOp>;
template class FC<fp16, SigmoidOp>;
template class FC<fp32, SigmoidOp>;
template class FC<fp64, SigmoidOp>;

template class FC<bf16, TanhOp>;
template class FC<fp16, TanhOp>;
template class FC<fp32, TanhOp>;
template class FC<fp64, TanhOp>;

template class FC<bf16, GELUOp>;
template class FC<fp16, GELUOp>;
template class FC<fp32, GELUOp>;
template class FC<fp64, GELUOp>;

template class FC<bf16, SwishOp>;
template class FC<fp16, SwishOp>;
template class FC<fp32, SwishOp>;
template class FC<fp64, SwishOp>;

}   //namespace alya
