#pragma once 

#include <cmath>
#include <cstddef>
#include <unordered_map>

#include <alya/optimizer/Optimizer.hpp>
#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya {

/// @brief AdamW optimizer | Updates paramters using adaptiv momentum
/// @tparam P Precison type (e.g. alya::fp32)
/// @param lr controls step size in optimizer updates
/// @param wd penalizes large weights by shrinking them during steps
/// @note lr: lower values improve stability but slows down convergence
/// @note wd: reduces overfitting
template <typename P>
class AdamW : public internal::optimizer<P> {
    using computeT = Precision<P>::computeT;

    struct State {
        Tensor<P, 2> mW, vW;         //m stands for momentum -> first moment, and v for variance -> second moment
        Tensor<P, 2> mB, vB;
        size_t t = 0;
        bool init = false;
    };

    std::unordered_map<Layer<P>*, State> state;

    computeT learningRate, beta1, beta2, epsilon;

public:
    AdamW(computeT learningRate, computeT weightDecay, computeT beta1 = computeT(0.9), computeT beta2 = computeT(0.999), computeT epsilon = computeT(1e-8)) : internal::optimizer<P>(learningRate, weightDecay), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    void step(Layer<P>& layer) override {
        if(layer.getWeights().device().type == DeviceType::CPU) {
            stepCpu(layer);
        } else {
            stepGpu(layer);
        }
    }

private:
    void stepCpu(Layer<P>& layer) {
        auto& s= state[&layer];

        if(!s.init) {
            //initialisation
            Device dev = layer.getWeights().device();

            s.mW = Tensor<P, 2>(layer.getWeights().numRows(), layer.getWeights().numCols(), dev);
            s.vW = Tensor<P, 2>(layer.getWeights().numRows(), layer.getWeights().numCols(), dev);
            s.mB = Tensor<P, 2>(layer.getBias().numRows(), layer.getBias().numCols(), dev);
            s.vB = Tensor<P, 2>(layer.getBias().numRows(), layer.getBias().numCols(), dev);

            if(dev.type == DeviceType::GPU) {
                s.mW.toGPU();
                s.vW.toGPU();
                s.mB.toGPU();
                s.vB.toGPU();
            }

            s.init = true;
        }

        s.t++;

        auto& dw = layer.getWeightsGradient();
        auto& db = layer.getBiasGradient();

        //m = beta1 * m + (1 - beta1) * g
        //v = beta2 * v + (1 - beta2) * g^2
        s.mW = s.mW.scalarScaleInplace(beta1).addInplace(dw.scalarScale(computeT(1) - beta1));
        s.vW = s.vW.scalarScaleInplace(beta2).addInplace(dw.square().scalarScale(computeT(1) - beta2));

        s.mB = s.mB.scalarScaleInplace(beta1).addInplace(db.scalarScale(computeT(1) - beta1));
        s.vB = s.vB.scalarScaleInplace(beta2).addInplace(db.square().scalarScale(computeT(1) - beta2));

        computeT b1t = computeT(1) - std::pow(beta1, s.t);
        computeT b2t = computeT(1) - std::pow(beta2, s.t);

        auto mhatW = s.mW.scalarScale(computeT(1) / b1t);
        auto vhatW = s.vW.scalarScale(computeT(1) / b2t);

        auto mhatB = s.mB.scalarScale(computeT(1) / b1t);
        auto vhatB = s.vB.scalarScale(computeT(1) / b2t);

        //update values     adamW: Wdecay = W - lr(mhat / (sqrt(vhat) + eps) + decay * W) --> is simplified --> original: W - lr(mhat / (sqrt(vhat) + eps)) - (W(lr * decay))
        layer.getWeights().subtractInplace(mhatW.divideInplace(vhatW.sqrtInplace().scalarAdd(epsilon)).scalarScale(this -> lr)).subtractInplace(layer.getWeights().scalarScale(this -> lr * this -> decay));
        layer.getBias().subtractInplace(mhatB.divideInplace(vhatB.sqrtInplace().scalarAdd(epsilon)).scalarScale(this -> lr));
    }

    void stepGpu(Layer<P>& layer) {
        stepCpu(layer);
    }
};

}   //namespace alya
