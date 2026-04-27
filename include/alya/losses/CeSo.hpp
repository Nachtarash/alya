#pragma once

#include <cstddef>
#include <cmath>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Cross Entropy Loss (softmax included) | Measures how well predicted class probabilities match the correct class
/// @tparam P Precison type (e.g. alya::fp32)
/// @note Formula: L = -Σ y_i * z_i + log(Σ e^z_j)
/// @note expects raw logits
template <typename P>
class CrossEntropyLoss : public internal::Loss<P> {
using computeT = Precision<P>::computeT;
using storageT = Precision<P>::storageT;
public:
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
        size_t classes = logits.numCols();

        const storageT* L = logits.cpuData();
        const storageT* Tgt = targets.cpuData();

        Tensor<P, 2> grad(batch, classes, logits.device());
        storageT* G = grad.cpuData();

        computeT loss = computeT(0);

        for(size_t i = 0; i < batch; i++) {
            computeT maxLogit = toCompute(L[i * classes]);

            for(size_t j = 1; j < classes; j++) {
                computeT otherLogit = toCompute(L[i * classes + j]);
                maxLogit = std::max(maxLogit, otherLogit);
            }

            computeT sumExp = computeT(0);
            for(size_t j = 0; j < classes; j++) {
                computeT otherExp = toCompute(L[i * classes + j]);
                sumExp  += std::exp(otherExp - maxLogit);
            }

            for(size_t j = 0; j < classes; j++) {
                computeT otherSofmaxVal = toCompute(L[i * classes + j]);
                computeT softmax = std::exp(otherSofmaxVal - maxLogit) / sumExp;
                computeT t = toCompute(Tgt[i * classes + j]);

                if(t == computeT(1)) {
                    loss -= std::log(softmax + computeT(1e-6));
                }
                
                G[i * classes + j] = toStorage<storageT>((softmax - t) / static_cast<computeT>(batch));
            }
        }

        loss /= static_cast<computeT>(batch);

        return { loss, grad };
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets);
};

}   //namespace alya