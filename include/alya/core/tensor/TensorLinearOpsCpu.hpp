#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <random>

#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>

namespace alya::TensorLinearOpsCpu {

template <typename X>
concept TensorLike = requires(X x, const X cx) {
    typename X::storageT;
    typename X::computeT;

    { cx.size() } -> std::convertible_to<size_t>;
    { x.cpuData() };
    { cx.cpuData() };
    { cx.emptyLike() } -> std::same_as<X>;
};

template <TensorLike TensorType>
TensorType hadamardCpu(const TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType C = A.emptyLike();

    const storageT* a = A.cpuData();
    const storageT* b = B.cpuData();
    storageT* c = C.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        c[i] = toStorage<storageT>(aVal * bVal);
    }
    
    return C;
}

template <TensorLike TensorType>
TensorType& hadamardInplaceCpu(TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();
    const storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        a[i] = toStorage<storageT>(aVal * bVal);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType divideCpu(const TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType C = A.emptyLike();

    const storageT* a = A.cpuData();
    const storageT* b = B.cpuData();
    storageT* c = C.cpuData();

    constexpr computeT eps = computeT(1e-6);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        c[i] = toStorage<storageT>(aVal * (bVal + eps));
    }
    
    return C;
}

template <TensorLike TensorType>
TensorType& divideInplaceCpu(TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();
    const storageT* b = B.cpuData();

    constexpr computeT eps = computeT(1e-6);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        a[i] = toStorage<storageT>(aVal * (bVal + eps));
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType addCpu(const TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType C = A.emptyLike();

    const storageT* a = A.cpuData();
    const storageT* b = B.cpuData();
    storageT* c = C.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        c[i] = toStorage<storageT>(aVal + bVal);
    }
    
    return C;
}

template <TensorLike TensorType>
TensorType& addInplaceCpu(TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();
    const storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        a[i] = toStorage<storageT>(aVal + bVal);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType subtractCpu(const TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType C = A.emptyLike();

    const storageT* a = A.cpuData();
    const storageT* b = B.cpuData();
    storageT* c = C.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        c[i] = toStorage<storageT>(aVal - bVal);
    }
    
    return C;
}

template <TensorLike TensorType>
TensorType& subtractInplaceCpu(TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();
    const storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);

        a[i] = toStorage<storageT>(aVal - bVal);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType scalarScaleCpu(const TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType B = A.emptyLike();

    const storageT* a = A.cpuData();
    storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        b[i] = toStorage<storageT>(aVal * scalar);
    }
    
    return B;
}

template <TensorLike TensorType>
TensorType& scalarScaleInplaceCpu(TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        a[i] = toStorage<storageT>(aVal * scalar);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType scalarDivideCpu(const TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType B = A.emptyLike();

    const storageT* a = A.cpuData();
    storageT* b = B.cpuData();

    constexpr computeT eps = computeT(1e-6);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        b[i] = toStorage<storageT>(aVal / (scalar + eps));
    }
    
    return B;
}

template <TensorLike TensorType>
TensorType& scalarDivideInplaceCpu(TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    constexpr computeT eps = computeT(1e-6);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        a[i] = toStorage<storageT>(aVal / (scalar + eps));
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType scalarAddCpu(const TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType B = A.emptyLike();

    const storageT* a = A.cpuData();
    storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        b[i] = toStorage<storageT>(aVal + scalar);
    }
    
    return B;
}

template <TensorLike TensorType>
TensorType& scalarAddInplaceCpu(TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        a[i] = toStorage<storageT>(aVal + scalar);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType scalarSubtractCpu(const TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType B = A.emptyLike();

    const storageT* a = A.cpuData();
    storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        b[i] = toStorage<storageT>(aVal - scalar);
    }
    
    return B;
}

template <TensorLike TensorType>
TensorType& scalarSubtractInplaceCpu(TensorType& A, const typename TensorType::computeT scalar) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        a[i] = toStorage<storageT>(aVal - scalar);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType squareCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType B = A.emptyLike();

    const storageT* a = A.cpuData();
    storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        b[i] = toStorage<storageT>(aVal * aVal);
    }
    
    return B;
}

template <TensorLike TensorType>
TensorType& squareInplaceCpu(TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        a[i] = toStorage<storageT>(aVal * aVal);
    }
    
    return A;
}

template <TensorLike TensorType>
TensorType sqrtCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    TensorType B = A.emptyLike();

    const storageT* a = A.cpuData();
    storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        b[i] = toStorage<storageT>(std::sqrt(aVal));
    }
    
    return B;
}

template <TensorLike TensorType>
TensorType& sqrtInplaceCpu(TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        a[i] = toStorage<storageT>(aVal * aVal);
    }
    
    return A;
}

template <TensorLike TensorType>
typename TensorType::storageT sumCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    computeT sum = computeT(0);

    const storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        sum += aVal;
    }

    return toStorage<storageT>(sum);
}

template <TensorLike TensorType>
typename TensorType::storageT meanCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    computeT sum = computeT(0);
    size_t total = A.size();

    const storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);

        sum += aVal;
    }

    return toStorage<storageT>(sum / total);
}

template <TensorLike TensorType>
typename TensorType::storageT minCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    const storageT* a = A.cpuData();

    computeT minVal = toCompute(a[0]);

    for(size_t i = 0; i < A.size(); i++) {
        computeT otherVal = toCompute(a[i]);
        if(otherVal < minVal) {
            minVal = otherVal;
        }
    }

    return toStorage<storageT>(minVal);
}

template <TensorLike TensorType>
typename TensorType::storageT maxCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    const storageT* a = A.cpuData();

    computeT maxVal = toCompute(a[0]);

    for(size_t i = 0; i < A.size(); i++) {
        computeT otherVal = toCompute(a[i]);
        if(otherVal > maxVal) {
            maxVal = otherVal;
        }
    }

    return toStorage<storageT>(maxVal);
}

template <TensorLike TensorType>
size_t argMinCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    const storageT* a = A.cpuData();

    size_t idx = 0;
    computeT minVal = toCompute(a[0]);

    for(size_t i = 0; i < A.size(); i++) {
        computeT otherVal = toCompute(a[i]);
        if(otherVal < minVal) {
            minVal = otherVal;
            idx = i;
        }
    }

    return idx;
}

template <TensorLike TensorType>
size_t argMaxCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    const storageT* a = A.cpuData();

    size_t idx = 0;
    computeT maxVal = toCompute(a[0]);

    for(size_t i = 0; i < A.size(); i++) {
        computeT otherVal = toCompute(a[i]);
        if(otherVal > maxVal) {
            maxVal = otherVal;
            idx = i;
        }
    }

    return idx;
}

template <TensorLike TensorType>
typename TensorType::storageT normCpu(const TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    const storageT* a = A.cpuData();

    computeT value = computeT(0);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        value += aVal * aVal;
    }

    return toStorage<storageT>(std::sqrt(value));
}

template <TensorLike TensorType>
TensorType& normalizeCpu(TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    computeT value = computeT(0);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        value += aVal * aVal;
    }

    value = std::sqrt(value);

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal2 = toCompute(a[i]);
        a[i] = toStorage<storageT>(aVal2 /= value);
    }

    return A;
}

template <TensorLike TensorType>
TensorType& fillZeroCpu(TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        a[i] = storageT(0);
    }

    return A;
}

template <TensorLike TensorType>
TensorType& fillOneCpu(TensorType& A) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        a[i] = storageT(1);
    }

    return A;
}

template <TensorLike TensorType>
TensorType& clipCpu(TensorType& A, typename TensorType::computeT minVal, typename TensorType::computeT maxVal) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        aVal = std::clamp(aVal, minVal, maxVal);
        a[i] = toStorage<storageT>(aVal);
    }

    return A;
}

template <TensorLike TensorType>
TensorType& initXavierCpu(TensorType& A, size_t fan) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    computeT limit = std::sqrt(computeT(6) / computeT(fan));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<computeT> dist(-limit, limit);

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        aVal = dist(gen);
        a[i] = toStorage<storageT>(aVal);
    }

    return A;
}

template <TensorLike TensorType>
TensorType& initHeCpu(TensorType& A, size_t fanIn) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    computeT stddev = std::sqrt(computeT(2) / computeT(fanIn));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<computeT> dist(computeT(0), stddev);

    storageT* a = A.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        aVal = dist(gen);
        a[i] = toStorage<storageT>(aVal);
    }

    return A;
}

template <TensorLike TensorType>
bool equalsCpu(const TensorType& A, const TensorType& B) {
    using storageT = typename TensorType::storageT;
    using computeT = typename TensorType::computeT;

    const storageT* a = A.cpuData();
    const storageT* b = B.cpuData();

    for(size_t i = 0; i < A.size(); i++) {
        computeT aVal = toCompute(a[i]);
        computeT bVal = toCompute(b[i]);
        if(aVal != bVal) {
            return false;
        }
    }

    return true;
}

}   //namespace alya::TensorLinearOpsCpu