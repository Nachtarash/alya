#pragma once

#include <concepts>
#include <cstddef>

#include <alya/core/ops/NumericalOps.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/activation/ActivationGpu.cuh>

namespace alya::TensorLinearOpsGpu {

template <typename X>
concept TensorLike = requires(X x, const X cx) {
    typename X::storageT;
    typename X::computeT;

    { cx.size() } -> std::convertible_to<size_t>;
    { cx.lastDim() } -> std::convertible_to<size_t>;
    { cx.outerDim() } -> std::convertible_to<size_t>;
    { x.gpuData() };
    { cx.gpuData() };
    { cx.emptyLike() } -> std::same_as<X>;
    { x.toGPU() };
};

template <TensorLike TensorType, typename Op>
TensorType activateGpu(const TensorType& A) {
    const size_t N = A.size();

    TensorType out = A.emptyLike();

    const_cast<TensorType&>(A).toGPU();
    out.toGPU();

    activationGpu::applyGpu<Op>(A.gpuData(), out.gpuData(), N);

    return out;
}

template <TensorLike TensorType, typename Op>
TensorType activationBackwardGpu(const TensorType& A, const TensorType& gradOut, const TensorType& a) {
    const size_t N = A.size();

    TensorType gradZ = A.emptyLike();

    const_cast<TensorType&>(A).toGPU();
    const_cast<TensorType&>(gradOut).toGPU();
    const_cast<TensorType&>(a).toGPU();
    gradZ.toGPU();

    activationGpu::backwardGpu<Op>(gradOut.gpuData(), A.gpuData(), a.gpuData(), gradZ.gpuData(), N);

    return gradZ;
}

template <TensorLike TensorType, typename Functor>
TensorType elementwise(const TensorType& A, const TensorType& B);

template <TensorLike TensorType, typename Functor>
TensorType& elementwiseInplace(TensorType& A, const TensorType& B);

template <TensorLike TensorType, typename Functor>
TensorType unary(const TensorType& A);

template <TensorLike TensorType, typename Functor>
TensorType& unaryInplace(TensorType& A); 

template <TensorLike TensorType, typename Functor>
TensorType scalar(const TensorType&, const typename TensorType::computeT scalar); 

template <TensorLike TensorType, typename Functor>
TensorType& scalarInplace(TensorType& A, const typename TensorType::computeT scalar);


template <TensorLike TensorType>
TensorType hadamardGpu(const TensorType& A, const TensorType& B) {
    return elementwise<TensorType, MulOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType& hadamardInplaceGpu(TensorType& A, const TensorType& B) {
    return elementwiseInplace<TensorType, MulOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType divideGpu(const TensorType& A, const TensorType& B) {
    return elementwise<TensorType, DivOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType& divideInplaceGpu(TensorType& A, const TensorType& B) {
    return elementwiseInplace<TensorType, DivOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType addGpu(const TensorType& A, const TensorType& B) {
    return elementwise<TensorType, AddOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType& addInplaceGpu(TensorType& A, const TensorType& B) {
    return elementwiseInplace<TensorType, AddOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType subtractGpu(const TensorType& A, const TensorType& B) {
    return elementwise<TensorType, SubOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType& subtractInplaceGpu(TensorType& A, const TensorType& B) {
    return elementwiseInplace<TensorType, SubOp<typename TensorType::storageT>>(A, B);
}

template <TensorLike TensorType>
TensorType squareGpu(const TensorType& A) {
    return unary<TensorType, SquareOp<typename TensorType::storageT>>(A);
}

template <TensorLike TensorType>
TensorType& squareInplaceGpu(TensorType& A) {
    return unaryInplace<TensorType, SquareOp<typename TensorType::storageT>>(A);
}

template <TensorLike TensorType>
TensorType sqrtGpu(const TensorType& A) {
    return unary<TensorType, SqrtOp<typename TensorType::storageT>>(A);
}

template <TensorLike TensorType>
TensorType& sqrtInplaceGpu(TensorType& A) {
    return unaryInplace<TensorType, SqrtOp<typename TensorType::storageT>>(A);
}

template <TensorLike TensorType>
TensorType scalarScaleGpu(const TensorType& A, const typename TensorType::computeT scalarV) {
    return scalar<TensorType, MulOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType& scalarScaleInplaceGpu(TensorType& A, const typename TensorType::computeT scalarV) {
    return scalarInplace<TensorType, MulOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType scalarDivideGpu(const TensorType& A, const typename TensorType::computeT scalarV) {
    return scalar<TensorType, DivOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType& scalarDivideInplaceGpu(TensorType& A, const typename TensorType::computeT scalarV) {
    return scalarInplace<TensorType, DivOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType scalarAddGpu(const TensorType& A, const typename TensorType::computeT scalarV) {
    return scalar<TensorType, AddOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType& scalarAddInplaceGpu(TensorType& A, const typename TensorType::computeT scalarV) {
    return scalarInplace<TensorType, AddOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType scalarSubtractGpu(const TensorType& A, const typename TensorType::computeT scalarV) {
    return scalar<TensorType, SubOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType& scalarSubtractInplaceGpu(TensorType& A, const typename TensorType::computeT scalarV) {
    return scalarInplace<TensorType, SubOp<typename TensorType::storageT>>(A, scalarV);
}

template <TensorLike TensorType>
TensorType multiplyLastAxisMaskGpu(const TensorType& A, const TensorType& mask);

template <TensorLike TensorType>
typename TensorType::storageT normGpu(const TensorType& A);

template <TensorLike TensorType>
TensorType& normalizeGpu(TensorType& A);

template <TensorLike TensorType>
TensorType& fillZeroGpu(TensorType& A);

template <TensorLike TensorType>
TensorType& fillOneGpu(TensorType& A);

template <TensorLike TensorType>
TensorType& clipGpu(TensorType& A, typename TensorType::computeT minVal, typename TensorType::computeT maxVal);

template <TensorLike TensorType>
typename TensorType::storageT sumGpu(const TensorType& A);

template <TensorLike TensorType>
typename TensorType::storageT meanGpu(const TensorType& A);

template <TensorLike TensorType>
typename TensorType::storageT minGpu(const TensorType& A);

template <TensorLike TensorType>
typename TensorType::storageT maxGpu(const TensorType& A);

template <TensorLike TensorType>
size_t argminGpu(const TensorType& A);

template <TensorLike TensorType>
size_t argmaxGpu(const TensorType& A);

}   //namespace alya::TensorLinearOpsGpu
