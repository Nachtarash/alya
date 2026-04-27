#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief KL divergence Loss | Measures the difference between a target probability distribution y and a predicted distribution p
/// @tparam P Precison type (e.g. alya::fp32)
/// @note Formula: L = Σ y_i * log(y_i / p_i)
template <typename P>
class klDivergenceLoss : public internal::Loss<P> {
public:
using storageT = typename Precision<P>::storageT;
using computeT = typename Precision<P>::computeT;

    internal::lossResult<P> forward(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) override {
        switch(logits.device().type) {
            case DeviceType::CPU:
                return forwardCpu(logits, targets);
            case DeviceType::GPU:
                return forwardGpu(logits, targets);
            default:
                throw std::runtime_error("Unknown device");
        }
    }

private:
    internal::lossResult<P> forwardCpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
        size_t batch = logits.numRows();
        size_t cols = logits.numCols();

        Tensor<P, 2> grad(batch, cols);
        computeT loss = computeT(0);
        constexpr computeT eps = computeT(1e-6);

        const storageT* L = logits.cpuData();
        const storageT* Tgt = targets.cpuData();
        storageT* G = grad.cpuData();

        for(size_t i = 0; i < batch; i++) {
            computeT sum_exp = computeT(0);

            computeT maxLogit = toCompute(L[i * cols]);
            for(size_t j = 1; j < cols; j++) {
                computeT otherLogit = toCompute(L[i * cols + j]);

                if(otherLogit > maxLogit) {
                    maxLogit = otherLogit;
                }
            }

            std::vector<computeT> softmax(cols);
            for(size_t j = 0; j < cols; j++) {
                computeT otherExp = toCompute(L[i * cols + j]);

                softmax[j] = std::exp(otherExp - maxLogit);
                sum_exp += softmax[j];
            }

            for(size_t j = 0; j < cols; j++) {
                softmax[j] /= sum_exp;

                computeT TgtVal = toCompute(Tgt[i * cols + j]);

                loss += TgtVal * std::log((TgtVal + eps) / (softmax[j] + eps));

                computeT GVal = toCompute(G[i * cols + j]);

                GVal = softmax[j] - TgtVal;

                G[i * cols + j] = toStorage<storageT>(GVal);
            }
        }

        loss /= batch;
        for(size_t i = 0; i < batch * cols; i++) {
            computeT GVal = toCompute(G[i]);

            GVal /= batch;

            G[i] = toStorage<storageT>(GVal);
        }

        return {loss, grad};
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets);
};

}   //namespace alya