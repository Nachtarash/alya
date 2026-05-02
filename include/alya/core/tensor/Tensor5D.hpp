#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <cstddef>


#include <alya/core/memory/TensorStorageBase.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/memory/Storage.hpp>

//NOT READY YET!!!

namespace alya {
    
/// @brief Tensor specialisation dim = 5
template <typename P>
class Tensor<P, 5> {
private:
    mutable Storage* storage;
    size_t dim0 = 0;
    size_t dim1 = 0;
    size_t dim2 = 0;
    size_t dim3 = 0;
    size_t dim4 = 0;

public: 
    Tensor() : storage(nullptr), dim0(0), dim1(0), dim2(0), dim3(0), dim4(0) {};
    Tensor(size_t d0, size_t d1, size_t d2, size_t d3, size_t d4, Device dev = Device{}) {};
};

}
