#pragma once

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya::internal {

template <typename P>
struct lossResult {
    using computeT = Precision<P>::computeT;
    computeT loss;
    Tensor<P, 2> grad;
};

template <typename P>
class Loss {
public:
    virtual lossResult<P> forward(const Tensor<P, 2>& logits, const Tensor<P, 2>& targets) = 0;

    virtual bool isClassification() const { return false; }; // default: regression

    virtual ~Loss() = default;
};

}   //namespace alya::internal
