#pragma once

#include <cstddef>

#include <alya/core/tensor/Tensorbase.hpp>
#include <alya/core/memory/Device.hpp>

namespace alya {

template <typename P, size_t inDim, size_t outDim>
class Layer {
public:
    using InputTensor = Tensor<P, inDim>;
    using OutputTensor = Tensor<P, outDim>;

    virtual ~Layer() = default;

    virtual OutputTensor forward(const InputTensor& input) = 0;
    virtual InputTensor backward(const OutputTensor& gradOut) = 0;

    virtual void setDevice(const Device& dev) = 0;

    virtual void setTraining(bool training) {};

    //virtual OutputTensor& getOutput() = 0;
};

}   //namespace alya
