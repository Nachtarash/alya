#pragma once 

#include <cstddef>
#include <type_traits>

#include <alya/core/tensor/Tensor.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/layer/LayerBase.hpp>
#include <alya/activation/ActivationCpu.hpp>
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
class FC : public Layer<P> {
private:
    using storageT = typename Precision<P>::storageT;
    using ActOpT = ActOp<storageT>;
 
    Tensor<P, 2> weights;
    Tensor<P, 2> bias;
    Tensor<P, 2> z;        //forward pre activation cache && reused as delta buffer in gpu backward
    Tensor<P, 2> a;        //forward cache
    Tensor<P, 2> input;    //backprop cache
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

    Tensor<P, 2>& getOutput() override { return a; }

    //parameter access for optimizers
    Tensor<P, 2>& getWeights() override { return weights; }
    Tensor<P, 2>& getBias() override { return bias; }
    Tensor<P, 2>& getWeightsGradient() override { return dw; }
    Tensor<P, 2>& getBiasGradient() override { return db; }

private:
    void initParams() {
        if constexpr(std::is_same_v<ActOpT, ReLuOp<storageT>> || std::is_same_v<ActOpT, LeakyReLuOp<storageT>> || std::is_same_v<ActOpT, ELUOp<storageT>>) {
            weights.initHe();
        } else {
            weights.initXavier();
        }

        bias.fillZero();
    }

    Tensor<P, 2> forwardCpu(const Tensor<P, 2>& inputIn) {
        input = inputIn.clone();

        z = input.matmul(weights).addBroadcastRow(bias);
        a = z.activate<ActOpT>();

        return a;
    }

    Tensor<P, 2> forwardGpu(const Tensor<P, 2>& inputIn);

    Tensor<P, 2> backwardCpu(const Tensor<P, 2>& gradOut) {
        Tensor<P, 2> dact = a.derivative<ActOpT>();
        Tensor<P, 2> delta = gradOut.clone();
        delta.hadamardInplace(dact);

        dw = input.transpose().matmul(delta);
        db = delta.sumRows();

        Tensor<P, 2> gradInput = delta.matmul(weights.transpose());
        return gradInput;
    }

    Tensor<P, 2> backwardGpu(const Tensor<P, 2>& gradOut);
};

}   //namespace alya