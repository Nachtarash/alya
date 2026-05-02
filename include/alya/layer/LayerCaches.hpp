#pragma once

#include <alya/activation/ActivationCache.hpp>

namespace alya {

template <typename TensorT>
struct FCCache {
    TensorT input;
    ActivationCache<TensorT> act;
};

}   //namespace alya
