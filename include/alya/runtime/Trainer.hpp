#pragma once

#include <iostream>
#include <iomanip>
#include <cstddef>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/data/Loader.hpp>
#include <alya/runtime/Validator.hpp>
#include <alya/layer/MLP.hpp>
#include <alya/losses/Loss.hpp>
#include <alya/optimizer/Optimizer.hpp>
#include <alya/profiling/Timer.hpp>

namespace alya {

/// @brief Trainer orchestrates model components
/// @tparam P Precison type (e.g. alya::fp32)
/// @param batchsize number of data processed per training step
/// @param inputDim size of data to start (e.g. 28x28 in mnist)
/// @param outputSize size of output dim (e.g. 10 in mnist because 10 classes) often classes etc in targets
/// @param model Model to train on
/// @param dataset dataset to train on (train)
/// @param valiset optinal parameter: dataset to validate on (test)
/// @param loss loss to calculate cost / how correct the model was
/// @param optimizer algorithm to updates parameters, the learing update methode
/// @param epochs amount of iterations to train on all data
/// @param verbose shows more detailed information about the training
/// @param device device on which the model should perform
template <typename P>
class Trainer {
using computeT = Precision<P>::computeT;

private:
    size_t batchsize;
    size_t inputDim;
    size_t targetDim;
    size_t epochs;
    bool verbose;
    bool validate = false;

    MLP<P>& model;
    loader<P>& dataset;
    loader<P>& valiset;
    internal::Loss<P>& loss;
    internal::optimizer<P>& opt;
    std::unique_ptr<Validator<P>> validator;

    Tensor<P, 2> input;
    Tensor<P, 2> target;

    Device device;

public:
    Trainer(
        size_t bs,
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        internal::Loss<P>& loss, 
        internal::optimizer<P>& o, 
        size_t e, 
        bool v = false, 
        const Device& dev = Device{})
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim), 
            epochs(e), 
            verbose(v), 
            model(m), 
            dataset(l), 
            loss(loss), 
            opt(o), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            device(dev) 
    {
        model.setDevice(dev);
    }

    Trainer(
        size_t bs, 
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        internal::Loss<P>& loss, 
        internal::optimizer<P>& o, 
        size_t e, 
        const Device& dev = Device{})
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim), 
            epochs(e), 
            model(m), 
            dataset(l), 
            loss(loss), 
            opt(o), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            device(dev) 
    {
        verbose = false;
        model.setDevice(dev);
    }

    Trainer(
        size_t bs, 
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        loader<P>& lv, 
        internal::Loss<P>& loss, 
        internal::optimizer<P>& o, 
        size_t e, 
        bool v = false, 
        const Device& dev = Device{}) 
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim),
            epochs(e), 
            verbose(v), 
            model(m), 
            dataset(l), 
            valiset(lv), 
            loss(loss), 
            opt(o), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            device(dev)
    {
        model.setDevice(dev);
        if(validate) {
            validator = std::make_unique<Validator<P>>(batchsize, inputDim, targetDim, model, valiset, loss, false, device);
        }
    }

    Trainer(
        size_t bs, 
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m,
        loader<P>& l, 
        loader<P>& lv, 
        internal::Loss<P>& loss, 
        internal::optimizer<P>& o, 
        size_t e, 
        const Device& dev = Device{})
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim), 
            epochs(e), 
            model(m),  
            dataset(l), 
            valiset(lv), 
            loss(loss), 
            opt(o), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            device(dev) 
    {
        verbose = false;
        model.setDevice(dev);
        validate = true;
        if(validate) {
            validator = std::make_unique<Validator<P>>(batchsize, inputDim, targetDim, model, valiset, loss, false, device);
        }
    }

    void train() {
        for(size_t epoch = 0; epoch < epochs; epoch++) {
            std::cout << "=== Epoch " << epoch + 1 << "/" << epochs << " ===" << std::endl;
            std::cout << "Loaded samples: " << dataset.dataSize() << " | Data per sample: " << inputDim << std::endl;

            size_t numBatches = (dataset.dataSize() + batchsize - 1) / batchsize;

            computeT epochLossSum = computeT(0);
            computeT epochAccSum = computeT(0);
            size_t batchesWithAcc = 0;
 
            double totalBatchTime = 0.0;

            Timer timerEpoch;
            timerEpoch.start();

            for(size_t i = 0; i < numBatches; i++) {
                Timer timerBatch;
                timerBatch.start();

                dataset.next(input, target);

                if(device.type == DeviceType::GPU) {
                    input.setDevice(device);
                    target.setDevice(device);
                    input.toGPU();
                    target.toGPU();
                }
                
                Tensor<P, 2> out = model.forward(input, true);
                internal::lossResult<P> res = loss.forward(out, target);

                epochLossSum += res.loss;

                //Acc for classifikation
                computeT batchAcc = computeT(0);

                if(loss.isClassification()) {
                    Tensor<P, 2> outAcc = out;
                    Tensor<P, 2> tgtAcc = target;
                    
                    if(device.type == DeviceType::GPU) {
                        outAcc = out.clone();
                        tgtAcc = target.clone();
                        outAcc.toCPU();
                        tgtAcc.toCPU();
                        Device cpu;
                        cpu.type = DeviceType::CPU;
                        outAcc.setDevice(cpu);
                        tgtAcc.setDevice(cpu);
                    }

                    size_t correct = 0;

                    for(size_t b = 0; b < batchsize; b++) {
                        size_t predClass = outAcc.argmaxRow(b);
                        size_t targetClass = tgtAcc.argmaxRow(b);

                        if(predClass == targetClass) {
                            correct++;
                        }
                    }

                    batchAcc = static_cast<computeT>(correct) / static_cast<computeT>(batchsize);
                    epochAccSum += batchAcc;
                    batchesWithAcc++;
                }

                model.backward(res.grad);
                model.step(opt);

                timerBatch.stop();
                totalBatchTime += timerBatch.getTime();
                double avgBatchTime = totalBatchTime / (i + 1);

                if(verbose) {
                    std::cout << "Batch " << i + 1 << "/" << numBatches << " | Loss: " << std::setprecision(6) << res.loss;
                    
                    if(loss.isClassification()) {
                        std::cout << " | Accuracy: " << std::setprecision(4) << batchAcc;
                    }

                    std::cout << " | Time: " << std::fixed << std::setprecision(3) << timerBatch.getTime() << "s" << " (avg: " << avgBatchTime << "s)" << std::endl;
                }

            }

            timerEpoch.stop();

            //epoch level statistik
            computeT epochLossAvg = epochLossSum / numBatches;
            computeT epochAccAvg = (batchesWithAcc > 0) ? (epochAccSum / static_cast<computeT>(batchesWithAcc)) : computeT(0);

            std::cout << "=== Epoch Summary ===" << std::endl;
            std::cout << "Avg Loss: " << std::setprecision(6) << epochLossAvg;

            if(loss.isClassification()) {
                std::cout << " |Avg Acc: " << std::setprecision(4) << epochAccAvg;
            }

            std::cout << " |Duration: " << std::fixed << std::setprecision(3) << timerEpoch.getTime() << "s" << std::endl;

            std::cout << std::endl;

            dataset.shuffle();

            if(validate && validator) {     //validator is always != nullptr if validate is true, it is basically redundant
                validator -> validate();
            }
        }
    }
};

}   //namespace alya