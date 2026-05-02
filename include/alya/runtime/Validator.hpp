#pragma once

#include <fstream>
#include <string>
#include <cstddef>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/data/Loader.hpp>
#include <alya/layer/MLP.hpp>
#include <alya/losses/Loss.hpp>
#include <alya/profiling/Timer.hpp>
#include <alya/profiling/ErrorMessage.hpp>
#include <alya/profiling/Print.hpp>

namespace alya {

/// @brief Validator orchestrates model components for only forward pass (testing)
/// @tparam P Precison type (e.g. alya::fp32)
/// @param batchsize number of data processed per training step
/// @param inputDim size of data to start (e.g. 28x28 in mnist)
/// @param outputSize size of output dim (e.g. 10 in mnist because 10 classes) often classes etc in targets
/// @param model Model to train on
/// @param dataset dataset to train on (train)
/// @param valiset optinal parameter: dataset to validate on (test)
/// @param loss loss to calculate cost / how correct the model was
/// @param epochs amount of iterations to train on all data
/// @param verbose shows more detailed information about the training
/// @param device device on which the model should perform
/// @note There are two optinal filepaths. These dont work in the moment -> UB or silent crash or just close
template <typename P>
class Validator {
using computeT = Precision<P>::computeT;
using storageT = Precision<P>::storageT;
private:
    size_t batchsize;
    size_t inputDim;
    size_t targetDim;
    std::string wFilename;
    std::string bFilename;
    bool verbose;
    bool loadFiles;

    MLP<P>& model;
    loader<P>& dataset;
    internal::Loss<P>& loss;

    Tensor<P, 2> input;
    Tensor<P, 2> target;

    Device device;

public:
    Validator(
        size_t bs, 
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        internal::Loss<P>& loss, 
        const std::string& wFile, 
        const std::string& bFile, 
        bool v = false, 
        const Device& dev = Device{})
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim), 
            model(m), 
            dataset(l), 
            loss(loss), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            wFilename(wFile), 
            bFilename(bFile), 
            verbose(v), 
            device(dev) 
    {
        loadFiles = true;
        model.setDevice(dev);
    }

    Validator(
        size_t bs, 
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        internal::Loss<P>& loss, 
        const std::string& wFile, 
        const std::string& bFile, 
        const Device& dev = Device{}) 
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim), 
            model(m), 
            dataset(l), 
            loss(loss), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            wFilename(wFile), 
            bFilename(bFile), 
            device(dev) 
    {
        verbose = false;
        loadFiles = true;
        model.setDevice(dev);
    }

    Validator(
        size_t bs, 
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        internal::Loss<P>& loss, 
        bool v = false, 
        const Device& dev = Device{}) 
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim), 
            model(m), 
            dataset(l), 
            loss(loss), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            verbose(v), 
            device(dev) 
    {
        loadFiles = false;
        model.setDevice(dev);
    }

    Validator(
        size_t bs,
        size_t inDim, 
        size_t tgtDim, 
        MLP<P>& m, 
        loader<P>& l, 
        internal::Loss<P>& loss, 
        const Device& dev = Device{})
        :   batchsize(bs), 
            inputDim(inDim), 
            targetDim(tgtDim),
            model(m), 
            dataset(l), 
            loss(loss), 
            input(bs, inDim), 
            target(bs, tgtDim), 
            device(dev) 
    {
        verbose = false;
        loadFiles = false;
        model.setDevice(dev);
    }

    void validate() {
        print("=== Validator ===\n");
        print("Loaded samples: ", dataset.dataSize(), " | Data per sample: ", inputDim, "\n");

        size_t numBatches = (dataset.dataSize() + batchsize - 1) / batchsize;
        computeT lossSum = computeT(0);
        computeT accSum = computeT(0);
        size_t batchsWithAcc = 0;
        double totalBatchTime = 0.0;
        Timer timerEpoch;
        timerEpoch.start();

        if(loadFiles) { //---------------- does not work right now -------------------------
            print("Loaded parameters from file");

            std::ifstream file_weights(wFilename, std::ios::binary);
            std::ifstream file_biases(bFilename, std::ios::binary);

            if(!file_weights.is_open() || !file_biases.is_open()) {
                ERRORMESSAGE("Validator: cannot open files for weights and/or biases!\n");
                std::exit(1);
            }

            for(auto* layer : model.getLayers()) {
                Tensor<P, 2>& weights = layer -> getWeights();
                Tensor<P, 2>& biases = layer -> getBias();

                size_t wSize = weights.size();
                size_t bSize = biases.size();

                storageT* weightsPtr = weights.cpuData();
                storageT* biasesPtr = biases.cpuData();

                std::vector<float> wBuffer(wSize);
                std::vector<float> bBuffer(bSize);

                file_weights.read(reinterpret_cast<char*>(wBuffer.data()), (wSize) * sizeof(float));
                file_biases.read(reinterpret_cast<char*>(bBuffer.data()), (bSize) * sizeof(float));

                //std::cout << "wSize = " << wSize << "\n";
                //std::cout << "expected bytes = " << wSize * sizeof(float) << "\n";
                //std::cout << "read bytes = " << file_weights.gcount() << "\n";

                for(size_t i = 0; i < wSize; i++) {
                    weightsPtr[i] = toStorage<storageT>(wBuffer[i]);
                }

                for(size_t i = 0; i < bSize; i++) {
                    biasesPtr[i] = toStorage<storageT>(bBuffer[i]);
                }
            }
        }   // ---------------------------------------------------------------------------------

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

            Tensor<P, 2> out = model.forward(input, false);
            internal::lossResult<P> result = loss.forward(out, target);

            lossSum += result.loss;
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
                accSum += batchAcc;
                batchsWithAcc++;
            } 

            timerBatch.stop();
            totalBatchTime += timerBatch.getTime();
            double avgBatchTime = totalBatchTime / (i + 1);

            if(verbose) {
                print("Batch ", i + 1, "/", numBatches, " | Loss: ", Fixed{6}, result.loss);

                if(loss.isClassification()) {
                    print(" | Acc: ", Fixed{4}, batchAcc);
                }

                print(" | Time: ", Fixed{3}, timerBatch.getTime(), "s (avg: ", avgBatchTime, "s)\n");
            }
        }

        timerEpoch.stop();

        computeT lossAvg = lossSum / numBatches;
        computeT accAvg = (batchsWithAcc > 0) ? (accSum / static_cast<computeT>(batchsWithAcc)) : computeT(0);

        print("=== Epoch Summary ===\n", "Avg Loss: ", Fixed{6}, lossAvg);

        if(loss.isClassification()) {
            print(" | Avg Acc: ", Fixed{4}, accAvg);
        }

        print(" | Duration: ", Fixed{3}, timerEpoch.getTime(), "s\n", Endl{});
    }
};

}   //namespace alya
