#pragma once

#include <cmath>
#include <vector>

#include <alya/core/data/Dataset.hpp>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/profiling/ErrorMessage.hpp>

namespace alya {

/// @brief Pre-defined Norm options
enum class NormType {
    NONE = 0,
    DIV255,
    ZERO_MEAN_STD__LOKAL,
    ZERO_MEAN_STD__GLOBAL
};

template <typename P>
class Normalization {
using computeT = Precision<P>::computeT;
private:
    NormType type;
    computeT global_mean = computeT(0);
    computeT global_std = computeT(1);

public:
    Normalization(NormType t = NormType::DIV255) : type(t) {}
    
    //compute global mean/std from all samples
    void computeGlobalStats(const internal::dataSet<P>& samples) {
        if(samples.data.empty()) {
            ERRORMESSAGE("samples have no data");
        }

        computeT mean = computeT(0);
        computeT m2 = computeT(0);
        size_t n = 0;

        for(computeT x : samples.data) {
            n++; 
            computeT delta = x - mean;
            mean += delta / n;
            computeT delta2 = x - mean;
            m2 += delta * delta2;
        }

        global_mean = mean;
        computeT variance = m2 / n;
        global_std = std::sqrt(variance);
        if(global_std == computeT(0)) { global_std = computeT(1); }
    }

    //Normalize samples
    void normalize(internal::dataSet<P>& samples) const {
        switch(type) {
            case NormType::NONE: break;
            case NormType::DIV255:
                for(size_t i = 0; i < samples.data.size(); i++) {
                    samples.data[i] /= computeT(255); 
                }

                break;

            case NormType::ZERO_MEAN_STD__LOKAL: {
                if(samples.inputDim == 0) {
                    ERRORMESSAGE("sample size is 0");    //inputdim is calculated in loader.hpp near : if(first sample)
                }

                size_t numSamples = samples.data.size() / samples.inputDim;

                for(size_t i = 0; i < numSamples; i++) {
                    size_t offset = i * samples.inputDim;

                    computeT mean = computeT(0);
                    computeT m2 = computeT(0);
                    size_t n = 0;

                    for(size_t j = 0; j < samples.inputDim; j++) {
                        computeT x = samples.data[offset + j];
                        n++;
                        computeT delta = x - mean;
                        mean += delta / n;
                        computeT delta2 = x - mean;
                        m2 += delta * delta2;
                    }

                    computeT variance = m2 / n;
                    computeT stddev = std::sqrt(variance);
                    if(stddev == computeT(0)) stddev = computeT(1);

                    for(size_t j = 0; j < samples.inputDim; j++) {
                        samples.data[offset + j] = (samples.data[offset + j] - mean) / stddev;
                    }
                }
                break;
            }

            case NormType::ZERO_MEAN_STD__GLOBAL:
                for(size_t i = 0; i < samples.data.size(); i++) {
                    samples.data[i] = (samples.data[i] - global_mean) / global_std;
                }
                break;

            default:
                ERRORMESSAGE("Unknown normilization type");
        }
    }
};

}   //namespace alya
