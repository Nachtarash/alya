#pragma once

#include <cstddef>
#include <cmath>

#include <alya/losses/Loss.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/memory/TensorStorageBase.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Cosine Similarity Loss | Measures similarity between prediction and target vectors based on angel
/// @tparam P Precison type (e.g. alya::fp32)
/// @note Formula: L = 1 - dot(x, y) / (|x| * |y|)
template <typename P>
class cosineSimilarityLoss : public internal::Loss<P> {
public:
using storageT = typename Precision<P>::storageT;
using computeT = typename Precision<P>::computeT;

    internal::lossResult<P> forward(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) override {
        return logits.deviceDispatcher(
            "cosineSimilarityForward",
            [&] { return forwardCpu(logits, targets); },
            [&] { return forwardGpu(logits, targets); });
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

        for(size_t i = 0; i < batch; i++) {
            computeT dot = computeT(0);
            computeT norm_pred = computeT(0);
            computeT norm_target = computeT(0);

            for(size_t j = 0; j < cols; j++) {
                computeT p = toCompute(L[i * cols + j]);
                computeT t = toCompute(Tgt[i * cols + j]);
                dot += p * t;
                norm_pred += p * p;
                norm_target += t * t;
            }
            
            norm_pred = std::sqrt(norm_pred);
            norm_target = std::sqrt(norm_target);

            if(norm_pred == computeT(0)) {
                norm_pred = computeT(1e-12);
            }

            if(norm_target == computeT(0)) {
                norm_target = computeT(1e-12);
            }

            computeT cos_sim = dot / (norm_pred * norm_target);
            loss += computeT(1) - cos_sim;

            for(size_t j = 0; j < cols; j++) {
                computeT p = toCompute(L[i * cols + j]);
                computeT t = toCompute(Tgt[i * cols + j]);
                computeT GVal = (t / (norm_pred * norm_target) - (dot / (norm_pred * norm_pred * norm_pred * norm_target)) * p) / (batch * cols);

                G[i * cols + j] = toStorage<storageT>(GVal);
            }
        }

        loss /= (batch * cols);
        return {loss, grad};
    }

    internal::lossResult<P> forwardGpu(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets);
};

}   //namespace alya
