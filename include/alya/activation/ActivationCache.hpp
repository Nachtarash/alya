#pragma once

namespace alya {

template <typename TensorT>
struct ActivationCache {
    TensorT z;
    TensorT a;
};

}   //namespace alya
