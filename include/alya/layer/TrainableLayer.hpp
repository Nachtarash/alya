#pragma once

#include <cstddef>

#include <alya/layer/LayerBase.hpp>
#include <alya/core/tensor/Tensorbase.hpp>

namespace alya {

template <typename P, size_t inDim, size_t outDim, size_t weightDim, size_t BiasDim = 2>
class TrainableLayer : public Layer<P, inDim, outDim> {
public:
    using WeightTensor = Tensor<P, weightDim>;
    using BiasTensor = Tensor<P, BiasDim>;
    using OutputTensor = Tensor<P, outDim>;

    virtual WeightTensor& getWeights() = 0;
    virtual BiasTensor& getBias() = 0;
    
    virtual WeightTensor& getWeightsGradient() = 0;
    virtual BiasTensor& getBiasGradient() = 0;

    virtual OutputTensor& getOutput() = 0;
};

}
