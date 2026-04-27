#pragma once

#include <vector>

#include <alya/layer/LayerBase.hpp>
#include <alya/layer/Dropout.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/optimizer/Optimizer.hpp>
#include <alya/losses/Loss.hpp> 

namespace alya {

/// @brief Multi layer perceptron. Stores Fc and dropout
/// @tparam Precison type (e.g. alya::fp32)
/// @note architecture is temporary und will very likly be changed
template <typename T>
class MLP {
private:
    bool training;

    std::vector<Layer<T>*> layers;
    std::vector<Dropout<T>*> drops;

public:
    const std::vector<Layer<T>*>& getLayers() const { return layers; }

    void setDevice(const Device& dev) {
        for(auto* layer : layers) {
            layer -> setDevice(dev);
        }
    }

    /// @brief adds an FC to MLP
    /// @param layer FC layer
    /// @attention Is temporary at its use at the moment/ not fully build for support all future layertypes
    void addLayer(Layer<T>* layer) {
        layers.push_back(layer);
    }

    /// @brief adds an Dropout to MLP
    /// @param layer Dropout layer
    /// @attention Is an temporary solution
    void addDropout(Dropout<T>* drop) {
        drops.push_back(drop);
    }

    /// @brief Performs forward-pass on all FC in MLP
    /// @param input Tensor with input-data
    /// @param isTraining flag for dropout to know wenn to compute mask
    /// @attention isTraining flag is not the final solution
    Tensor<T, 2> forward(const Tensor<T, 2>& input, bool isTraining) {
        Tensor<T, 2> out = input;
        training = isTraining;

        for(auto* layer : layers) {
            out = layer -> forward(out);

            for(auto* d : drops) {
                if(d -> getTargetLayer() == layer) {
                    d -> forward(isTraining);
                }
            }
        }

        return out;
    }

    /// @brief Performs backward-pass on all Fc in MLP
    /// @param gradOutput Tensor with gradientsOutput
    void backward(const Tensor<T, 2>& gradOutput) {
        Tensor<T, 2> grad = gradOutput;

        for(int i = layers.size() - 1; i >= 0; i--) {
            for(auto* d : drops) {
                if(d -> getTargetLayer() == layers[i]) {
                    d -> backward(grad);
                }
            }

            grad = layers[i] -> backward(grad);
        }
    }

    /// @brief Performs weights/bias update
    /// @param opt Optimizer who should perform the updates
    void step(internal::optimizer<T>& opt) {
        for(auto& layer : layers) {
            opt.step(*layer);
        }
    }
};

}      //namespace alya