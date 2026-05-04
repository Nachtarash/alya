#pragma once

#include <cstddef>
#include <random>

#include <alya/core/memory/Device.hpp>
#include <alya/core/tensor/Tensor.hpp>
#include <alya/layer/layerBase.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya {

/// @brief Art of normalization | creates a mask with values 0 and 1 and deactivates the Neuron elementwise
/// @tparam P Precison type (e.g. alya::fp32)
/// @tparam Dim dimension from layer (e.g. fc has dim of 2)
/// @param rate rate in percent to deactivate Neurons (e.g. name(0.02))
template <typename P, size_t Dim>
class Dropout : public Layer<P, Dim, Dim> {
private:
    using computeT = Precision<P>::computeT;
    using storageT = Precision<P>::storageT;

    using TensorT = Tensor<P, Dim>;

    bool isTraining = true;
    computeT dropoutRate;

    TensorT mask;

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<computeT> dist{0.0, 1.0};

public:
    Dropout(computeT rate) : dropoutRate(rate) {}

    void setTraining(bool training) override {
        isTraining = training;
    }

    void setDevice(const Device& dev) override {
        if(mask.size() == 0) {
            return;
        }

        mask.setDevice(dev);
    }

    /// @brief Performs dropout
    TensorT forward(const TensorT& input) override {
        if(!isTraining) {
            return input.clone();
        }

        mask = input.emptyLike();

        initMask();

        return input.hadamard(mask);
    }

    TensorT backward(const TensorT& gradOut) override {
        if(!isTraining) {
            return gradOut.clone();
        }

        return gradOut.hadamard(mask);
    }

private:
    void initMask() {
        const computeT scale = computeT(1) / (computeT(1) - dropoutRate);
        storageT* data = mask.cpuData();

        for(size_t i = 0; i < mask.size(); i++) {
            const computeT randVal = dist(gen);
            const computeT value = (randVal < dropoutRate) ? computeT(0) : scale;
            data[i] = toStorage<storageT>(value);
        }
    }
};

}   //naemspace alya
