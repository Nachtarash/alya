#pragma once

#include <vector>

#include <alya/layer/LayerBase.hpp>
#include <alya/layer/TrainableLayer.hpp>
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
    std::vector<Layer<T, 2, 2>*> layers;
    std::vector<TrainableLayer<T, 2, 2, 2, 2>*> trainableLayers;

public:
    const std::vector<TrainableLayer<T, 2, 2, 2, 2>*>& getTrainableLayers() const { return trainableLayers; }
    const std::vector<Layer<T, 2, 2>*>& getLayers() const { return layers; }

    void setDevice(const Device& dev) {
        for(auto* layer : layers) {
            layer -> setDevice(dev);
        }
    }

    /// @brief adds Layer to MLP
    /// @param layer Non trainable layer
    void addLayer(Layer<T, 2, 2>* layer) {
        layers.push_back(layer);
    }

    void addTrainableLayer(TrainableLayer<T, 2, 2, 2, 2>* trainablelayer) {
        layers.push_back(trainablelayer);
        trainableLayers.push_back(trainablelayer);
    }

    /// @brief Performs forward-pass on all FC in MLP
    /// @param input Tensor with input-data
    /// @param isTraining flag for dropout to know wenn to compute mask
    Tensor<T, 2> forward(const Tensor<T, 2>& input, bool isTraining) {
        Tensor<T, 2> out = input;

        for(auto* layer : layers) {
            layer -> setTraining(isTraining);
            
            out = layer -> forward(out);
        }

        return out;
    }

    /// @brief Performs backward-pass on all Fc in MLP
    /// @param gradOutput Tensor with gradientsOutput
    void backward(const Tensor<T, 2>& gradOutput) {
        Tensor<T, 2> grad = gradOutput;

        for(int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i] -> backward(grad);
        }
    }

    /// @brief Performs weights/bias update
    /// @param opt Optimizer who should perform the updates
    void step(internal::optimizer<T>& opt) {
        for(auto& layer : trainableLayers) {
            opt.step(*layer);
        }
    }
};

}      //namespace alya
