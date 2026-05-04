#pragma once 

#include <alya/layer/LayerBase.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya::internal {

template <typename P>
class optimizer {
using computeT = Precision<P>::computeT;

protected:
    computeT lr;
    computeT decay;

public:
    optimizer(computeT learningRate, computeT weightDecay) : lr(learningRate), decay(weightDecay) {}
    virtual ~optimizer() = default;
    
    virtual void step(TrainableLayer<P, 2, 2, 2, 2>& layer) = 0;
};

}   //namespace alya::internal
