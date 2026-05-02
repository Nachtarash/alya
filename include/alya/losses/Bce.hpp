#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/memory/TensorStorageBase.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Binary Cross Entropy Loss (sigmoid included) | Measures how well predicted probabilities match binary targets E {0, 1}
/// @tparam P Precison type (e.g. alya::fp32)
/// @note Formula: L = -[y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
/// @note expects raw logits
template <typename P>
class bceLoss : public internal::Loss<P> {
public:
using storageT = typename Precision<P>::storageT;
using computeT = typename Precision<P>::computeT;

    internal::lossResult<P> forward(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) override {
        return logits.deviceDispatcher(
            "bceForward",
            [&] { return forwardCpu(logits, targets); },
            [&] { return forwardGpu(logits, targets); });
    }

    bool isClassification() const override { return true; }

private:
    internal::lossResult<P> forwardCpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) {
        size_t batch = logits.numRows();
        size_t cols = logits.numCols();

        Tensor<P, 2> grad(batch, cols, logits.device());
        computeT loss = computeT(0);

        const storageT* L = logits.cpuData();
        const storageT* Tgt = targets.cpuData();
        storageT* G = grad.cpuData();

        constexpr computeT eps = computeT(1e-6);

        for(size_t i = 0; i < batch * cols; i++) {
            computeT x = toCompute(L[i]);
            computeT y = toCompute(Tgt[i]);

            computeT sigmoid;
            if(x >= 0) {
                computeT z = std::exp(-x);
                sigmoid = computeT(1) / (computeT(1) + z);
            } else {
                computeT z = std::exp(x);
                sigmoid = z / (computeT(1) + z);
            }

            loss += -(y * std::log(sigmoid + eps) + (computeT(1) - y) * std::log(computeT(1) - sigmoid + eps));
            computeT GVal = (sigmoid - y) / batch;

            G[i] = toStorage<storageT>(GVal);
        }

        loss /= batch;
        return { loss, grad };
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& target);
};

}      //namespace alya
