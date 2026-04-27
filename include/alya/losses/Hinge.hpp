#pragma once

#include <cstddef>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Hinge Loss | Penelizes prediction that are incorrect or within the margin
/// @tparam P Precison type (e.g. alya::fp32)
/// @note Formula: L = max(0, 1 - y * f(x))
/// @note expects logits in {-1, 1}
template <typename P>
class hingeLoss : public internal::Loss<P> {
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

    bool isClassification() const override { return true; }

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
                computeT p = toCompute(L[i]);
                computeT t = toCompute(Tgt[i]);
                computeT diff = computeT(1) - p * t;

                if(diff > 0) {
                    loss += diff;

                    storageT GVal = toStorage<storageT>(-t / (batch * cols));

                    G[i] = GVal;
                } else {
                    G[i] = gpuZero<storageT>();
                }
        }

        loss /= (batch * cols);
        return {loss, grad};
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets);
};

}   //namespace alya