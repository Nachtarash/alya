#pragma once

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>

namespace alya {

template <typename P>
class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor<P, 2> forward(const Tensor<P, 2>& input) = 0;
    virtual Tensor<P, 2> backward(const Tensor<P, 2>& gradOut) = 0;

    virtual void setDevice(const Device& dev) = 0;

    virtual Tensor<P, 2>& getOutput() = 0;

    virtual Tensor<P, 2>& getWeights() = 0;
    virtual Tensor<P, 2>& getBias() = 0;
    virtual Tensor<P, 2>& getWeightsGradient() = 0;
    virtual Tensor<P, 2>& getBiasGradient() = 0;
};

}   //namespace alya