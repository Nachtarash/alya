#pragma once 

#include <cstdint>
#include <cstddef>
#include <vector>

#include <alya/core/precision/PrecisonTypes.cuh>

namespace alya::internal {

template <typename P>
struct dataSet {
    using computeT = Precision<P>::computeT;
    std::vector<uint32_t> labels;
    std::vector<computeT> data;
    size_t inputDim;
};

}   //namespace alya::internal