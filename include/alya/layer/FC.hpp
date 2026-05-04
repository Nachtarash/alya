#pragma once 

#include <cstddef>
#include <type_traits>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/layer/TrainableLayer.hpp>
#include <alya/activation/ActivationCpu.hpp>
#include <alya/layer/LayerCaches.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya {

/// @brief Fully connected layer (y = f(Wx + b))
/// @tparam P Precison type (e.g. alya::fp32)
/// @tparam actOp Activation (e.g. ReLuOp)
/// @param inputSize featuresIn dimension
/// @param outputSize featuresOut dimension
/// @note Info and/or if manuell use, not using buildIn Trainer:
/// @note Input shape: [batchsize, featuresIn]
/// @note Output shape: [batchsize, featuresOut]
template <typename P, template <typename> class ActOp>
class FC : public TrainableLayer<P, 2, 2, 2, 2> {
private:
    using storageT = typename Precision<P>::storageT;
    using ActOpT = ActOp<storageT>;
 
    Tensor<P, 2> weights;
    Tensor<P, 2> bias;
    
    FCCache<Tensor<P, 2>> cache;

    Tensor<P, 2> dw;       //gradients weights
    Tensor<P, 2> db;       //gradients biases

public:
    FC(size_t inputSize, size_t outputSize) : weights(inputSize, outputSize), bias(1, outputSize) {
        initParams();
    }

    Tensor<P, 2> forward(const Tensor<P, 2>& inputIn) override {
        if(inputIn.device().type == DeviceType::CPU) {
            return forwardCpu(inputIn);
        } else {
            return forwardGpu(inputIn);
        }
    }

    Tensor<P, 2> backward(const Tensor<P, 2>& grad_out) override {
        if(grad_out.device().type == DeviceType::CPU) {
            return backwardCpu(grad_out);
        } else {
            return backwardGpu(grad_out);
        }
    }

    void setDevice(const Device& dev) override {
        weights.setDevice(dev);
        bias.setDevice(dev);

        if(dev.type == DeviceType::GPU) {
            weights.toGPU();
            bias.toGPU();
        } else {  
            weights.toCPU();
            bias.toCPU();
        }
    }

    Tensor<P, 2>& getOutput() override { return cache.act.a; }

    //parameter access for optimizers
    Tensor<P, 2>& getWeights() override { return weights; }
    Tensor<P, 2>& getBias() override { return bias; }
    Tensor<P, 2>& getWeightsGradient() override { return dw; }
    Tensor<P, 2>& getBiasGradient() override { return db; }

private:
    void initParams() {
        if constexpr(std::is_same_v<ActOpT, ReLuOp<storageT>> || 
                     std::is_same_v<ActOpT, LeakyReLuOp<storageT>> || 
                     std::is_same_v<ActOpT, ELUOp<storageT>> ||
                     std::is_same_v<ActOpT, GELUOp<storageT>> ||
                     std::is_same_v<ActOpT, SwishOp<storageT>>)
                    {
            weights.initHe();
        } else {
            weights.initXavier();
        }

        bias.fillZero();
    }

    Tensor<P, 2> forwardCpu(const Tensor<P, 2>& inputIn) {
        cache.input = inputIn.clone();

        cache.act.z = cache.input.matmul(weights).addBroadcastRow(bias);
        cache.act.a = cache.act.z.template activate<ActOpT>();

        return cache.act.a;
    }

    Tensor<P, 2> forwardGpu(const Tensor<P, 2>& inputIn);

    Tensor<P, 2> backwardCpu(const Tensor<P, 2>& gradOut) {
        Tensor<P, 2> gradZ = cache.act.z.template activationBackward<ActOpT>(gradOut, cache.act.a);

        dw = cache.input.transpose().matmul(gradZ);
        db = gradZ.sumRows();

        Tensor<P, 2> gradInput = gradZ.matmul(weights.transpose());
        return gradInput;
    }

    Tensor<P, 2> backwardGpu(const Tensor<P, 2>& gradOut);
};

}   //namespace alya
