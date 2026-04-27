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
#include <alya/core/tensor/Tensorbase.hpp>

//NOT READY YET!!!

namespace alya {

/// @brief Tensor specialisation dim = 3
template <typename T>
class Tensor<T, 3> : private internal::TensorStorageBase<T> {
private:
    size_t dim0 = 0;
    size_t dim1 = 0;
    size_t dim2 = 0;

public:
    using internal::TensorStorageBase<T>::storage;
    using internal::TensorStorageBase<T>::device;
    using internal::TensorStorageBase<T>::setDevice;
    using internal::TensorStorageBase<T>::toCPU;
    using internal::TensorStorageBase<T>::toGPU;

    Tensor() = default;
    Tensor(size_t d0, size_t d1, size_t d2, Device dev = Device{}) : internal::TensorStorageBase<T>(d0 * d1 * d2 * sizeof(T), dev), dim0(d0), dim1(d1), dim2(d2) {};

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;

    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    ~Tensor() = default;

    size_t numDim0() const {return dim0; }
    size_t numDim1() const {return dim1; }
    size_t numDim2() const {return dim2; }

    inline size_t offset(size_t i, size_t j, size_t k) const {
        return (i * dim1 * dim2) + (j * dim2) + k;
    }

    inline size_t size() const { return dim0 * dim1 * dim2; }

    Tensor emptyLike() const { return Tensor(dim0, dim1, dim2, device()); }
};

}