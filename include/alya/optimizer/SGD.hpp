#pragma once

#include <alya/optimizer/Optimizer.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya {

/// @brief Stochastic Gradient Descent optimizer | Updates paramters using gradient descent with optinal weight decay
/// @tparam P Precison type (e.g. alya::fp32)
/// @param lr controls step size in optimizer updates
/// @param wd penalizes large weights by shrinking them during steps
/// @note lr: lower values improve stability but slows down convergence
/// @note wd: reduces overfitting
template <typename P>
class SGD : public internal::optimizer<P> {
using computeT = Precision<P>::computeT;

public:
    SGD(computeT learning_rate, computeT weight_decay = computeT(0)) : internal::optimizer<P>(learning_rate, weight_decay) {}

    void step(Layer<P>& layer) override {
        if(layer.getWeights().device().type == DeviceType::CPU) {
            stepCpu(layer);
        } else {
            stepGpu(layer);
        }
    }

private:
    void stepCpu(Layer<T>& layer) {
        layer.getWeights().subtractInplace(layer.getWeightsGradient().add(layer.getWeights().scalarScale(this -> decay)).scalarScale(this -> lr));       //Wdecay = W - lr(dW + (decay * W))
        layer.getBias().subtractInplace(layer.getBiasGradient().scalarScale(this -> lr));
    }

    void stepGpu(Layer<T>& layer) {
        layer.getWeights().subtractInplace(layer.getWeightsGradient().add(layer.getWeights().scalarScale(this -> decay)).scalarScale(this -> lr));      //Wdecay = W - lr(dW + (decay * W))
        layer.getBias().subtractInplace(layer.getBiasGradient().scalarScale(this -> lr));
    }
};

}   //namespace alya
