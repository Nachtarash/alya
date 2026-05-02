#pragma once 

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <alya/core/tensor/Tensor2D.hpp>
#include <alya/core/data/Normalization.hpp>
#include <alya/core/data/Dataset.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/profiling/ErrorMessage.hpp>

namespace alya::internal {

template <typename T>
T strToT(const std::string& s);

template <>
inline float strToT<float>(const std::string& s) {
    return std::stof(s);
}

template <>
inline double strToT<double>(const std::string& s) {
    return std::stod(s);
}

uint32_t readInt(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) |
           (uint32_t(bytes[3])); 
}

}   //namespace alya::internal

namespace alya {

/// @brief Pre-defined loader-options
enum LoaderType {
    NONE,
    MNIST,
    EMNISTDIGITS,
    EMNISTBALANCED,
    EMNISTBYMERGE,
    EMNISTBYCLASS,
    EMNISTLETTERS
};

/// @brief Loads data into an DataSet
/// @tparam P Precison type (e.g. alya::fp32)
/// @param vFilename FilePath to feature-data
/// @param lFilename FilePath to label-data | only when labels are seperate
/// @param batch batchsize
/// @param type loaderType
template <typename P>
class loader{
using computeT = Precision<P>::computeT;
using storageT = Precision<P>::storageT;
private:
    internal::dataSet<P> dataset;
    size_t batchSize;
    size_t cursor = 0;
    LoaderType selection;

public:
    loader(const std::string& filename, size_t batch, LoaderType type) : batchSize(batch), selection(type) {
        switch(type) {
            case MNIST:
                singleLoaderMnist(filename, batch);
                break;
            case EMNISTDIGITS: 
                ERRORMESSAGE("Not available");
            case EMNISTLETTERS: 
                ERRORMESSAGE("Not available");
            case NONE:
                ERRORMESSAGE("Not available");
            default:
                ERRORMESSAGE("Invalid LoaderType");
        }
    }
    
    loader(const std::string& vFilename, const std::string& lFilename, size_t batch, LoaderType type) : batchSize(batch), selection(type) {
        switch(type) {
            case MNIST: 
                dualLoaderEmnist(vFilename, lFilename, batch);
                break;
            case EMNISTDIGITS: 
                dualLoaderEmnist(vFilename, lFilename, batch);
                break;
            case EMNISTBALANCED: 
                dualLoaderEmnist(vFilename, lFilename, batch);
                break;
            case EMNISTBYMERGE: 
                dualLoaderEmnist(vFilename, lFilename, batch);
                break;
            case EMNISTBYCLASS: 
                dualLoaderEmnist(vFilename, lFilename, batch);
                break;
            case EMNISTLETTERS: 
                dualLoaderEmnistLetters(vFilename, lFilename, batch);
                break;
            case NONE:
                dualLoaderNew(vFilename, lFilename, batch);
                break;
            default:
                ERRORMESSAGE("Invalid LoaderType");
        };
    }

    void shuffle() {
        static std::mt19937 rng(std::random_device{}());
    
        size_t numSamples = dataset.data.size() / dataset.inputDim;
        size_t dim = dataset.inputDim;

        std::vector<size_t> perm(numSamples);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<computeT> newData(dataset.data.size());
        std::vector<uint32_t> newLabels(numSamples);

        for(size_t i = 0; i < numSamples; i++) {
            size_t origIdx = perm[i];
            size_t newStartIdx = i * dim;
            size_t origStartIdx = origIdx * dim;

            std::copy(dataset.data.begin() + origStartIdx, dataset.data.begin() + origStartIdx + dim, newData.begin() + newStartIdx);

            newLabels[i] = dataset.labels[origIdx];
        }

        dataset.data.swap(newData);
        dataset.labels.swap(newLabels);
    }

    /// @brief Splits DataSet and loads DataSet (Input, Target)
    /// @param input Tensor
    /// @param target Tensor
    void next(Tensor<P, 2>& input, Tensor<P, 2>& target) {
        if(input.numRows() != batchSize || target.numRows() != batchSize) {
            ERRORMESSAGE("Invalid batchsize");
        }

        storageT* inPtr = input.cpuData();
        storageT* tgtPtr = target.cpuData();

        for(size_t i = 0; i < batchSize; i++) {
            size_t idx = (cursor + i) % dataset.labels.size();

            for(size_t k = 0; k < dataset.inputDim; k++) {
                inPtr[i * dataset.inputDim + k] = toStorage<storageT>(dataset.data[idx * dataset.inputDim + k]);
            }

            uint32_t label = dataset.labels[idx];
            for(size_t j = 0; j < target.numCols(); j++) {
                tgtPtr[i * target.numCols() + j] = (j == label) ? gpuOne<storageT>() : gpuZero<storageT>();
            }
        }

        cursor = (cursor + batchSize) % dataset.labels.size();      //if divide by bachtsize has an rest amount of 0 it beginns again from the start
    }

    /// @brief Normalizes dataSet
    /// @param type Norm type (e.g. alya::NormType::DIV55)
    void normalize(NormType type) {
        Normalization<P> norm(type);

        if(type == NormType::ZERO_MEAN_STD__GLOBAL) {
            norm.computeGlobalStats(dataset);
        }

        norm.normalize(dataset);
    }

    size_t dataSize() const {
        size_t numSamples = (dataset.inputDim == 0) ? 0 : dataset.data.size() / dataset.inputDim;

        return numSamples;
    }

private:
    void singleLoaderMnist(const std::string& filename, size_t batch) {
        std::ifstream file(filename);
        if(!file.is_open()) { 
            ERRORMESSAGE("file couldn't be opend");
        }

        std::string line;
        bool firstSample = true;

        while(std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;

            uint32_t label;
            std::vector<computeT> pixels;
            bool first = true;

            while(std::getline(ss, value, ',')) {
                computeT v = internal::strToT<computeT>(value);

                if(first) {
                    label = static_cast<uint32_t>(v);
                    first = false;
                } else {
                    pixels.push_back(v);
                }
            }

            if(firstSample) {
                dataset.inputDim = pixels.size();
                if(pixels.size() != dataset.inputDim) { 
                    ERRORMESSAGE("sample size mismatch"); 
                }

                firstSample = false;
            }

            dataset.labels.push_back(label);
            dataset.data.insert(dataset.data.end(), pixels.begin(), pixels.end());
        }
    }

    void dualLoaderEmnist(const std::string& vFilename, const std::string& lFilename, size_t batch) {
        std::ifstream v_file(vFilename, std::ios::binary);
        if(!v_file.is_open()) {
            ERRORMESSAGE("couldn't be opend: " + vFilename);
        }

        std::ifstream l_file(lFilename, std::ios::binary);
        if(!l_file.is_open()) { 
            ERRORMESSAGE("couldn't be opend: " + lFilename);
        }

        uint32_t vMagic = internal::readInt(v_file);
        uint32_t numImages = internal::readInt(v_file);
        uint32_t rows = internal::readInt(v_file);
        uint32_t cols = internal::readInt(v_file);

        uint32_t lMagic = internal::readInt(l_file);
        uint32_t numLabels = internal::readInt(l_file);

        if(vMagic != 2051 || lMagic != 2049) {
            ERRORMESSAGE("invalid IDX magic number");
        }
        if(numImages != numLabels) {
            ERRORMESSAGE("image/label count mismatch");
        }

        dataset.inputDim = static_cast<size_t>(rows) * static_cast<size_t>(cols);

        dataset.data.resize(dataset.inputDim * static_cast<size_t>(numImages));
        dataset.labels.resize(numLabels);

        std::vector<uint8_t> tmp(dataset.data.size());
        v_file.read(reinterpret_cast<char*>(tmp.data()), tmp.size());

        std::vector<uint8_t> tmpLabels(dataset.labels.size());
        l_file.read(reinterpret_cast<char*>(tmpLabels.data()), tmpLabels.size());

        for(size_t i = 0; i < tmp.size(); i++) {
            dataset.data[i] = static_cast<computeT>(tmp[i]);
        }

        for(size_t i = 0; i < tmpLabels.size(); i++) {
            dataset.labels[i] = static_cast<uint32_t>(tmpLabels[i]);
        }
    }

    void dualLoaderEmnistLetters(const std::string& vFilename, const std::string& lFilename, size_t batch) {
        std::ifstream v_file(vFilename, std::ios::binary);
        if(!v_file.is_open()) {
            ERRORMESSAGE("couldn't be opend: " + vFilename);
        }

        std::ifstream l_file(lFilename, std::ios::binary);
        if(!l_file.is_open()) {
            ERRORMESSAGE("couldn't be opend: " + lFilename);
        }

        uint32_t vMagic = internal::readInt(v_file);
        uint32_t numImages = internal::readInt(v_file);
        uint32_t rows = internal::readInt(v_file);
        uint32_t cols = internal::readInt(v_file);

        uint32_t lMagic = internal::readInt(l_file);
        uint32_t numLabels = internal::readInt(l_file);

        if(vMagic != 2051 || lMagic != 2049) {
            ERRORMESSAGE("invalid IDX magic number");
        }
        if(numImages != numLabels) {
            ERRORMESSAGE("image/label count mismatch");
        }

        dataset.inputDim = static_cast<size_t>(rows) * static_cast<size_t>(cols);

        dataset.data.resize(dataset.inputDim * static_cast<size_t>(numImages));
        dataset.labels.resize(numLabels);

        std::vector<uint8_t> tmp(dataset.data.size());
        v_file.read(reinterpret_cast<char*>(tmp.data()), tmp.size());

        std::vector<uint8_t> tmpLabels(dataset.labels.size());
        l_file.read(reinterpret_cast<char*>(tmpLabels.data()), tmpLabels.size());

        for(size_t i = 0; i < tmp.size(); i++) {
            dataset.data[i] = static_cast<computeT>(tmp[i]);
        }

        for(size_t i = 0; i < tmpLabels.size(); i++) {
            if(tmpLabels[i] == 0) {
                ERRORMESSAGE("expected labels should be in range 1-26");
            }

            dataset.labels[i] = static_cast<uint32_t>(tmpLabels[i] - 1);
        }
    }

    void dualLoaderNew(const std::string& vFilename, const std::string& lFilename, size_t batch) {
        ERRORMESSAGE("NO loader made!!!");
    }
};

}   //namespace alya
