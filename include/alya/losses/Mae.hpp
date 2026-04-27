#pragma once

#include <cstddef>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Mean Absolute Error Loss | Measures the average absolute difference between prediction and target
/// @tparam P Precison type (e.g. alya::fp32)
/// @note Formula: L = (1/n) Σ |x_i - y_i|
template <typename P>
class maeLoss : public internal::Loss<P> {
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

        const storageT* L = logits.cpuData();
        const storageT* Tgt = targets.cpuData();
        storageT* G = grad.cpuData();

        for(size_t i = 0; i < batch * cols; i++) {
            computeT LVal = toCompute(L[i]);
            computeT TgtVal = toCompute(Tgt[i]);

            computeT diff = LVal - TgtVal;

            computeT GVal = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) / (batch * cols);

            G[i] = toStorage<storageT>(GVal);
            loss += fabs(diff);
        }

        loss /= (batch * cols);
        return {loss, grad};
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets);
};

}   //namespace alya