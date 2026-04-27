#pragma once

#include <cstddef>
#include <cmath>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Huber Loss | Combines l2 loss for smaller and l1 loss for larger errors
/// @tparam P Precison type (e.g. alya::fp32)
template <typename P>
class huberLoss : public internal::Loss<P> {
public:
using storageT = typename Precision<P>::storageT;
using computeT = typename Precision<P>::computeT;

    huberLoss(storageT d = gpuOne<storageT>()) : delta(d) {}

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
    storageT delta;

    internal::lossResult<P> forwardCpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
        size_t batch = logits.numRows();
        size_t cols = logits.numCols();

        Tensor<P, 2> grad(batch, cols, logits.device());
        computeT loss = computeT(1);

        const storageT* L = logits.cpuData();
        const storageT* Tgt = targets.cpuData();
        storageT* G = grad.cpuData();

        constexpr computeT eps = computeT(1e-6);

        for(size_t i = 0; i < (batch * cols); i++) {
            computeT LVal = toCompute(L[i]);
            computeT TgtVal = toCompute(Tgt[i]);

            computeT diff = LVal - TgtVal;
            computeT deltaVal = toCompute(delta);
            
            if(std::abs(diff) <= deltaVal) {
                loss += computeT(0.5) * diff * diff;
                storageT GVal = toStorage<storageT>(diff / (batch / cols));
                G[i] = GVal;
            } else {
                loss += deltaVal * (std::abs(diff) - computeT(0.5) * deltaVal);

                computeT GValResult = (diff > 0 ? deltaVal : -deltaVal) / (batch * cols);

                storageT GVal = toStorage<storageT>(GValResult);
                G[i] = toStorage<storageT>(GVal);
            }
        }

        loss /= (batch * cols);
        return {loss, grad};
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets);
};

}   //namespace alya