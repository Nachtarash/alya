#pragma once

#include <cstddef>
#include <random>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/layer/layerBase.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya {

/// @brief Art of normalization | creates a mask with values 0 and 1 and deactivates the Neuron for the layer
/// @tparam P Precison type (e.g. alya::fp32)
/// @param layer layer to perform dropout on
/// @param rate rate in percent to deactivate Neurons (e.g. name(layerX, 0.02))
template <typename P>
class Dropout {
private:
    using computeT = Precision<P>::computeT;
    using storageT = Precision<P>::storageT;

    computeT dropoutRate;

    Tensor<P, 2> mask;
    Layer<P>& target;

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<computeT> dist{0.0, 1.0};

public:
    Dropout(Layer<P>& l, computeT rate) : target(l), dropoutRate(rate) {}

    /// @brief Performs dropout on layer, targeted in constuctor
    /// @param isTraining flag for Training/Validaton
    void forward(bool isTraining) {
        if(!isTraining) {
            mask = Tensor<P, 2>();
            return;
        } 
        
        Tensor<P, 2>& out = target.getOutput();
        mask = Tensor<P, 2>(1, out.numCols());

        initMask();

        out.multiplyBroadcastRowInplace(mask);
    }

    void backward(Tensor<P, 2>& grad) {
        grad.multiplyBroadcastRowInplace(mask);
    }

    Layer<P>* getTargetLayer() { return &target; }

private:
    void initMask() {
        for(size_t i = 0; i < mask.numCols(); i++) {
            computeT randVal = dist(gen);

            if(randVal < dropoutRate) {
                mask.cpuData()[i] = gpuZero<storageT>();
            } else {
                mask.cpuData()[i] = gpuOne<storageT>() / (gpuOne<storageT>() - toStorage<storageT>(dropoutRate));
            }
        }
    }
};

}   //naemspace alya
