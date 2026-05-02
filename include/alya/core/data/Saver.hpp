#pragma once
 
#include <ostream>
#include <cstddef>
#include <string>
#include <vector>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/layer/FC.hpp>
#include <alya/layer/MLP.hpp>

namespace alya {

template <typename P>
void saveModel(const MLP<P>& mlp, const std::string& wFilename, const std::string& bFilename) {
    std::ofstream fileWeights(wFilename, std::ios::binary);
    std::ofstream fileBiases(bFilename, std::ios::binary);

    for(auto* layer : mlp.layers) {
        using computeT = Precision<P>::computeT;
        using storageT = Precision<P>::storageT;

        Tensor<P, 2>& weights = layer -> getWeights();
        Tensor<P, 2>& biases = layer -> getBias();

        const storageT* weightsPtr = weights.cpuData();
        const storageT* biasesPtr = biases.cpuData();

        size_t wSize = weights.size();
        size_t bSize = biases.size();

        std::vector<float> wBuffer(wSize);
        std::vector<float> bBuffer(bSize);

        for(size_t i = 0; i < wSize; i++) {
            wBuffer[i] = static_cast<float>(toCompute(weightsPtr[i]));
        }

        for(size_t i = 0; i < bSize; i++) {
           bBuffer[i] = static_cast<float>(toCompute(biasesPtr[i]));
        }

        fileWeights.write(reinterpret_cast<const char*>(wBuffer.data()), (wSize) * sizeof(float));
        fileBiases.write(reinterpret_cast<const char*>(bBuffer.data()), (bSize) * sizeof(float));
    }
}

/*template <typename T>
void saveModelTxt(const MLP<T>& mlp, const std::string& wFilename, const std::string& bFilename) {
    std::ofstream fileWeights(wFilename, std::ios::out);
    std::ofstream fileBiases(bFilename, std::ios::out);
    
    for(auto* layer : mlp.layers) {
        Tensor2D<T>& weights = layer -> getWeights();
        Tensor2D<T>& biases = layer -> getBias();

        T* wPtr = weights.cpuData();
        T* bPtr = biases.cpuData();

        for(size_t i = 0; i < weights.numRows() * weights.numCols(); i++) {
            fileWeights << wPtr[i] << " ";
        }

        fileWeights << "\n";

        for(size_t i = 0; i < biases.numRows() * biases.numCols(); i++) {
            fileBiases << bPtr[i] << " ";
        }

        fileBiases << "\n";
    }
}*/

}   //namespace alya
