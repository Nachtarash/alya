#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cmath>

#include <alya/core/memory/TensorStorageBase.hpp>
#include <alya/core/tensor/Tensorbase.hpp>
#include <alya/core/tensor/TensorLinearOpsCpu.hpp>
#include <alya/core/tensor/TensorLinearOpsGpu.hpp>
#include <alya/core/tensor/Tensoraxis.hpp>
#include <alya/core/memory/Device.hpp>
#include <alya/core/precision/PrecisonTypes.cuh>
#include <alya/core/precision/PrecisionUtils.cuh>
#include <alya/activation/ActivationCpu.hpp>
#include <alya/activation/ActivationGpu.cuh>
#include <alya/profiling/Print.hpp>

namespace alya {

/// @brief Tensor specialisation dim = 2
template <typename P>
class Tensor<P, 2> : private internal::TensorStorageBase<typename Precision<P>::storageT> {
private:
    size_t rows = 0;
    size_t cols = 0;

public:
    using storageT = typename Precision<P>::storageT;
    using computeT = typename Precision<P>::computeT;

    using internal::TensorStorageBase<typename Precision<P>::storageT>::storage;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::device;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::setDevice;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::printDevice;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::deviceDispatcher;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::toCPU;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::toGPU;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::cpuData;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::gpuData;
    using internal::TensorStorageBase<typename Precision<P>::storageT>::TensorClone;

    Tensor() = default;
    Tensor(size_t r, size_t c, Device dev = Device{}) : internal::TensorStorageBase<storageT>(r * c * sizeof(storageT), dev), rows(r), cols(c) {}

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;

    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    ~Tensor() = default;

    inline size_t numRows() const { return rows; }
    inline size_t numCols() const { return cols; }

    inline size_t offset(size_t r, size_t c) const { return r * cols + c; }

    inline size_t size() const { return rows * cols; }

    inline size_t lastDim() const { return numCols(); }

    inline size_t outerDim() const { return numRows(); }

    void BYTES() const { 
       size_t bytes = rows * cols * sizeof(storageT);
        print("BYTES: ", bytes, Endl{});
    }

    Tensor emptyLike() const { return Tensor(rows, cols, device()); }
    
    Tensor clone() const {
        return TensorClone(*this, emptyLike());
    }

    /// @brief Tensor(x, cols)
    Tensor getRow(size_t rowIdx) {
        Tensor row(1, cols);

        const storageT* a = cpuData();
        storageT* r = row.cpuData();

        for(size_t j = 0; j < cols; j++) {
            computeT val = toCompute(a[offset(rowIdx, j)]);
            r[j] = toStorage<storageT>(val);
        }

        return row;
    }

    /// @brief print Tensor
    void printTensor() const {
        const storageT* data = cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                computeT val = toCompute(data[offset(i, j)]);

                print(val, " ");
            }

            print(Endl{});
        }
    }

    /// @brief f(Tensor)
    template <typename Op>
    Tensor activate() const {
        return deviceDispatcher(
            "activate",
            [&] { return TensorLinearOpsCpu::activateCpu<Tensor, Op>(*this); },
            [&] { return TensorLinearOpsGpu::activateGpu<Tensor, Op>(*this); });
    }

    /// @brief gradOut * f'(Tensor)
    template <typename Op>
    Tensor activationBackward(const Tensor& gradOut, const Tensor& a) const {
        return deviceDispatcher(
            "activationBackward",
            [&] { return TensorLinearOpsCpu::activationBackwardCpu<Tensor, Op>(*this, gradOut, a); },
            [&] { return TensorLinearOpsGpu::activationBackwardGpu<Tensor, Op>(*this, gradOut, a); });
    }

    //for gpu implementation
    template <typename Functor, TensorAxis axis>
    Tensor broadcast(Tensor& B) const;

    //for gpu implementation
    template <typename Functor, TensorAxis axis>
    Tensor& broadcastInplace(const Tensor& B);


    /// @brief Tensor(M, K) * Tensor(K, N)
    Tensor matmul(const Tensor& B) const {
        return deviceDispatcher(
            "matmul", 
            [&] { return matmulCpu(B); }, 
            [&] { return matmulGpu(B); });
    }

    /// @brief Tensor * Tensor 
    Tensor hadamard(const Tensor& B) const {
        return deviceDispatcher(
            "hadamard", 
            [&] { return hadamardCpu(B); }, 
            [&] { return hadamardGpu(B); });
    }
    
    /// @brief Tensor * Tensor -> inplace
    Tensor& hadamardInplace(const Tensor& B) {
        return deviceDispatcher(
            "hadamardInplace", 
            [&]() -> Tensor& { return hadamardInplaceCpu(B); }, 
            [&]() -> Tensor& { return hadamardInplaceGpu(B); });
    }

    /// @brief Tensor / Tensor
    Tensor divide(const Tensor& B) const {
        return deviceDispatcher(
            "divide", 
            [&] { return divideCpu(B); }, 
            [&] { return divideGpu(B); });
    }

    /// @brief Tensor / Tensor -> inplace
    Tensor& divideInplace(const Tensor& B) {
        return deviceDispatcher(
            "divideInplace", 
            [&]() -> Tensor& { return divideInplaceCpu(B); }, 
            [&]() -> Tensor& { return divideInplaceGpu(B); });
    }

    /// @brief Tensor + Tensor
    Tensor add(const Tensor& B) const {
        return deviceDispatcher(
            "add", 
            [&] { return addCpu(B); }, 
            [&] { return addGpu(B); });
    }

    /// @brief Tensor + Tensor -> inplace
    Tensor& addInplace(const Tensor& B)  {
        return deviceDispatcher(
            "addInplace", 
            [&]() -> Tensor& { return addInplaceCpu(B); },
            [&]() -> Tensor& { return addInplaceGpu(B); });
    }

    /// @brief Tensor - Tensor
    Tensor subtract(const Tensor& B) const {
        return deviceDispatcher(
            "subtract", 
            [&] { return subtractCpu(B); }, 
            [&] { return subtractGpu(B); });
    }

    /// @brief Tensor - Tensor -> inplace
    Tensor& subtractInplace(const Tensor& B) {
        return deviceDispatcher(
            "subtractInplace", 
            [&]() -> Tensor& { return subtractInplaceCpu(B); }, 
            [&]() -> Tensor& { return subtractInplaceGpu(B); });
    }

    /// @brief Tensor * scalar
    Tensor scalarScale(const computeT scalar) const {
        return deviceDispatcher(
            "scalarScale", 
            [&] { return scalarScaleCpu(scalar); }, 
            [&] { return scalarScaleGpu(scalar); });
    }

    /// @brief Tensor * scalar -> inplace
    Tensor& scalarScaleInplace(const computeT scalar) {
        return deviceDispatcher(
            "scalarScaleInplace", 
            [&]() -> Tensor& { return scalarScaleInplaceCpu(scalar); }, 
            [&]() -> Tensor& { return scalarScaleInplaceGpu(scalar); });
    }
    
    /// @brief Tensor / scalar 
    Tensor scalarDivide(const computeT scalar) const {
        return deviceDispatcher(
            "scalarDivide", 
            [&] { return scalarDivideCpu(scalar); }, 
            [&] { return scalarDivideGpu(scalar); });
    }
    
    /// @brief Tensor / scalar -> inplace
    Tensor& scalarDivideInplace(const computeT scalar) {
        return deviceDispatcher(
            "scalarDivideInplace", 
            [&]() -> Tensor& { return scalarDivideInplaceCpu(scalar); }, 
            [&]() -> Tensor& { return scalarDivideInplaceGpu(scalar); });
    }

    /// @brief Tensor + scalar 
    Tensor scalarAdd(const computeT scalar) const {
        return deviceDispatcher(
            "scalarAdd", 
            [&] { return scalarAddCpu(scalar); }, 
            [&] { return scalarAddGpu(scalar); });
    }
    
    /// @brief Tensor + scalar -> inplace
    Tensor& scalarAddInplace(const computeT scalar) {
        return deviceDispatcher(
            "scalarAddInplace", 
            [&]() -> Tensor& { return scalarAddInplaceCpu(scalar); }, 
            [&]() -> Tensor& { return scalarAddInplaceGpu(scalar); });
    }

    /// @brief Tensor - scalar
    Tensor scalarSubtract(const computeT scalar) const {
        return deviceDispatcher(
            "scalarSubtract", 
            [&] { return scalarSubtractCpu(scalar); }, 
            [&] { return scalarSubtractGpu(scalar); });
    }
    
    /// @brief Tensor - scalar -> inplace
    Tensor& scalarSubtractInplace(const computeT scalar) {
        return deviceDispatcher(
            "scalarSubtractInplace", 
            [&]() -> Tensor& { return scalarSubtractInplaceCpu(scalar); }, 
            [&]() -> Tensor& { return scalarSubtractInplaceGpu(scalar); });
    }

    /// @brief Tensor ^2   
    Tensor square() const {
        return deviceDispatcher(
            "square", 
            [&] { return squareCpu(); }, 
            [&] { return squareGpu(); });
    }
    
    /// @brief Tensor ^2  -> inplace
    Tensor& squareInplace() {
        return deviceDispatcher(
            "squareInplace", 
            [&]() -> Tensor& { return squareInplaceCpu(); }, 
            [&]() -> Tensor& { return squareInplaceGpu(); });
    }

    /// @brief Tensor sqrt
    Tensor sqrt() const {
        return deviceDispatcher(
            "sqrt", 
            [&] { return sqrtCpu(); }, 
            [&] { return sqrtGpu(); });
    }

    /// @brief Tensor sqrt -> inplace
    Tensor& sqrtInplace() {
        return deviceDispatcher(
            "sqrtInplace", 
            [&]() -> Tensor& { return sqrtInplaceCpu(); }, 
            [&]() -> Tensor& { return sqrtInplaceGpu(); });
    }

    /// @brief Tensor + Tensor(1, cols)
    Tensor addBroadcastRow(Tensor& B) {
        return deviceDispatcher(
            "addBroadcastRow", 
            [&] { return addBroadcastRowCpu(B); }, 
            [&] { return addBroadcastRowGpu(B); });
    }

    /// @brief Tensor * Tensor(1, cols)
    Tensor multiplyBroadcastRow(Tensor& B) {
        return deviceDispatcher(
            "multiplyBroadcastRow", 
            [&] { return multiplyBroadcastRowCpu(B); }, 
            [&] { return multiplyBroadcastRowGpu(B); });
    }

    /// @brief Tensor * Tensor(1, cols) -> inplace
    Tensor& multiplyBroadcastRowInplace(const Tensor& B) {
        return deviceDispatcher(
            "multiplyBroadcastRowInplace", 
            [&]() -> Tensor& { return multiplyBroadcastRowInplaceCpu(B); }, 
            [&]() -> Tensor& { return multiplyBroadcastRowInplaceGpu(B); });
    }
    
    /// @brief Tensor + Tensor(rows, 1)
    Tensor addBroadcastCol(Tensor& B) {
        return deviceDispatcher(
            "addBroadcastCol", 
            [&] { return addBroadcastColCpu(B); }, 
            [&] { return addBroadcastColGpu(B); });
    }

    /// @brief Tensor * Tensor(rows, 1)
    Tensor multiplyBroadcastCol(Tensor& B) {
        return deviceDispatcher(
            "multiplyBroadcastCol", 
            [&] { return multiplyBroadcastColCpu(B); }, 
            [&] { return multiplyBroadcastColGpu(B); });
    }

    Tensor multiplyLastAxisMask(const Tensor& mask) const {
        return deviceDispatcher(
            "multiplyLastAxisMask",
            [&] { return multiplyLastAxisMaskCpu(mask); },
            [&] { return multiplyLastAxisMaskGpu(mask); });
    }

    /// @brief Σ Tensor -> x
    storageT sum() const {
        return deviceDispatcher(
            "sum", 
            [&] { return sumCpu(); }, 
            [&] { return sumGpu(); });
    }

    /// @brief Σ Tensor -> Tensor(1, cols)
    Tensor sumRows() const {
        return deviceDispatcher(
            "sumRows", 
            [&] { return sumRowsCpu(); }, 
            [&] { return sumRowsGpu(); });
    }

    /// @brief Σ Tensor -> Tensor(rows, 1)
    Tensor sumCols() const {
        return deviceDispatcher(
            "sumCols", 
            [&] { return sumColsCpu(); }, 
            [&] { return sumColsGpu(); });
    }

    /// @brief Σ Tensor(x, cols) -> x
    storageT sumRow(size_t row) const {
        return deviceDispatcher(
            "sumRow", 
            [&] { return sumRowCpu(row); }, 
            [&] { return sumRowCpu(row); });    //no gpu version
    }

    /// @brief Σ Tensor(rows, x) -> x
    storageT sumCol(size_t col) const {
        return deviceDispatcher(
            "sumCol", 
            [&] { return sumColCpu(col); }, 
            [&] { return sumColCpu(col); });    //no gpu version
    }

    /// @brief Σ Tensor / n -> x
    storageT mean() const {
        return deviceDispatcher(
            "mean", 
            [&] { return meanCpu(); }, 
            [&] { return meanGpu(); });
    }

    /// @brief min(Tensor) -> x
    storageT min() const {
        return deviceDispatcher(
            "min", 
            [&] { return minCpu(); }, 
            [&] { return minGpu(); });
    }

    /// @brief min(Tensor) -> Tensor(1, cols)
    Tensor minRows() const {
        return deviceDispatcher(
            "minRows", 
            [&] { return minRowsCpu(); }, 
            [&] { return minRowsGpu(); });
    }

    /// @brief max(Tensor) -> x
    storageT max() const {
        return deviceDispatcher(
            "max", 
            [&] { return maxCpu(); }, 
            [&] { return maxGpu(); });
    }

    /// @brief max(Tensor) -> Tensor(1, cols)
    Tensor maxRows() const {
        return deviceDispatcher(
            "maxRows", 
            [&] { return maxRowsCpu(); }, 
            [&] { return maxRowsGpu(); });
    }

    /// @brief argmin(Tensor) -> idx
    size_t argmin() const {
        return deviceDispatcher(
            "argmin", 
            [&] { return argminCpu(); }, 
            [&] { return argminGpu(); });
    }

    /// @brief argmax(Tensor(x, cols)) -> idx x
    size_t argminRow(size_t row) const {
        return deviceDispatcher(
            "argminRow", 
            [&] { return argminRowCpu(row); }, 
            [&] { return argminRowCpu(row); });     //no gpu version
    }

    /// @brief argmin(Tensor) -> idxTensor(1, cols)
    Tensor<size_t, 2> argminRows() const {
        return deviceDispatcher(
            "argminRows", 
            [&] { return argminRowsCpu(); }, 
            [&] { return argminRowsGpu(); });
    }

    /// @brief argmax(Tensor) -> idx
    size_t argmax() const {
        return deviceDispatcher(
            "argmax", 
            [&] { return argmaxCpu(); }, 
            [&] { return argmaxGpu(); });
    }

    /// @brief argmax(Tensor(x, cols)) -> idx x
    size_t argmaxRow(size_t row) const {
        return deviceDispatcher(
            "argmaxRow", 
            [&] { return argmaxRowCpu(row); }, 
            [&] { return argmaxRowCpu(row); });     //no gpu version
    }

    /// @brief argmax(Tensor) -> idxTensor(1, cols)
    Tensor<size_t, 2> argmaxRows() const {
        return deviceDispatcher(
            "argmaxRows", 
            [&] { return argmaxRowsCpu(); }, 
            [&] { return argmaxRowsGpu(); });
    }

    /// @brief L2 norm (frobenius norm) 
    storageT norm() const {
        return deviceDispatcher(
            "norm", 
            [&] { return normCpu(); }, 
            [&] { return normGpu(); });
    }
  
    /// @brief Tensor / |Tensor| -> 1
    Tensor& normalize() {
        return deviceDispatcher(
            "normalize", 
            [&]() -> Tensor& { return normalizeCpu(); }, 
            [&]() -> Tensor& { return normalizeGpu(); });
    }

    /// @brief Tensor(M, N) -> Tensor(N, M)
    Tensor transpose() const {
        return deviceDispatcher(
            "transpose", 
            [&] { return transposeCpu(); }, 
            [&] { return transposeGpu(); });
    }

    /// @brief Tensor = 0
    Tensor& fillZero() {
        return deviceDispatcher(
            "fillZero", 
            [&]() -> Tensor& { return fillZeroCpu(); }, 
            [&]() -> Tensor& { return fillZeroGpu(); });
    }

    /// @brief Tensor = 1
    Tensor& fillOne() {
        return deviceDispatcher(
            "fillOne", 
            [&]() -> Tensor& { return fillOneCpu(); }, 
            [&]() -> Tensor& { return fillOneGpu(); });
    }

    /// @brief clamp(Tensor, min, max)
    Tensor& clip(computeT minVal, computeT maxVal) {
        return deviceDispatcher(
            "clip", 
            [&]() -> Tensor& { return clipCpu(minVal, maxVal); }, 
            [&]() -> Tensor& { return clipGpu(minVal, maxVal); });
    }

    /// @brief Tensor(rows, cols) -> Tensor(newRows, newCols)
    Tensor& reshape(size_t newrows, size_t newcols) {
        return deviceDispatcher(
            "reshape", 
            [&]() -> Tensor& { return reshapeCpu(newrows, newcols); }, 
            [&]() -> Tensor& { return reshapeCpu(newrows, newcols); });     //no gpu version
    }

    /// @brief Tensor(rows, cols) -> Tensor(1, cols)
    Tensor& flatten() {
        return deviceDispatcher(
            "flatten", 
            [&]() -> Tensor& { return reshapeCpu(1, cols); }, 
            [&]() -> Tensor& { return reshapeCpu(1, cols); });       //no gpu version
    }

    /// @brief W ~ N(0, sqrt(2/fanIn)) Tanh/Sigmoid
    Tensor& initXavier() {
        return deviceDispatcher(
            "initXavier", 
            [&]() -> Tensor& { return initXavierCpu(); }, 
            [&]() -> Tensor& { return initXavierCpu(); });       //no gpu version
    }

    /// @brief W ~ N(0, sqrt(2/(fanIn + fanOut))) ReLu/LeakyReLu/ELU
    Tensor& initHe() {
        return deviceDispatcher(
            "initHe", 
            [&]() -> Tensor& { return initHeCpu(); }, 
            [&]() -> Tensor& { return initHeCpu(); });      //no gpu version
    }

    /// @brief if(Tensor A == Tensor B)
    bool equals(const Tensor& B) const {
        return deviceDispatcher(
            "equals", 
            [&] { return equalsCpu(B); }, 
            [&] { return equalsCpu(B); });      //no gpu version
    }

private:
    Tensor matmulCpu(const Tensor& B) const {
        assert(cols == B.rows);

        Tensor C(rows, B.cols);

        const storageT* a = cpuData();
        const storageT* b = B.cpuData();
        storageT* c = C.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < B.cols; j++) {

                computeT sum = computeT(0);

                for(size_t k = 0; k < cols; k++) {
                    computeT aVal = toCompute(a[offset(i, k)]); 
                    computeT bVal = toCompute(b[B.offset(k, j)]);

                    sum += aVal * bVal;
                }

                c[C.offset(i, j)] = toStorage<storageT>(sum);
            }
        }

        return C;
    }

    Tensor matmulGpu(const Tensor& B) const;

    Tensor hadamardCpu(const Tensor& B) const {
        assert(rows == B.rows && cols == B.cols);

        return TensorLinearOpsCpu::hadamardCpu<Tensor>(*this, B);
    }

    Tensor hadamardGpu(const Tensor& B) const {
        return TensorLinearOpsGpu::hadamardGpu<Tensor>(*this, B);
    }
    
    Tensor& hadamardInplaceCpu(const Tensor& B) {
        assert(rows == B.rows && cols == B.cols);

        return TensorLinearOpsCpu::hadamardInplaceCpu<Tensor>(*this, B);
    }

    Tensor& hadamardInplaceGpu(const Tensor& B) {
        return TensorLinearOpsGpu::hadamardInplaceGpu<Tensor>(*this, B);
    }

    Tensor divideCpu(const Tensor& B) const {
        assert(rows == B.rows && cols == B.cols);

        return TensorLinearOpsCpu::divideCpu<Tensor>(*this, B);
    }

    Tensor divideGpu(const Tensor& B) const {
        return TensorLinearOpsGpu::divideGpu<Tensor>(*this, B);
    }
    
    Tensor& divideInplaceCpu(const Tensor& B) {
        assert(rows == B.rows && cols == B.cols);

        return TensorLinearOpsCpu::divideInplaceCpu<Tensor>(*this, B);        
    }

    Tensor& divideInplaceGpu(const Tensor& B) {
        return TensorLinearOpsGpu::divideInplaceGpu<Tensor>(*this, B);
    }

    Tensor addCpu(const Tensor& B) const {
        assert(cols == B.cols && rows == B.rows);

        return TensorLinearOpsCpu::addCpu<Tensor>(*this, B);
    }

    Tensor addGpu(const Tensor& B) const {
        return TensorLinearOpsGpu::addGpu<Tensor>(*this, B);
    }
    
    Tensor& addInplaceCpu(const Tensor& B) {
        assert(rows == B.rows && cols == B.cols);

        return TensorLinearOpsCpu::addInplaceCpu<Tensor>(*this, B);
    }

    Tensor& addInplaceGpu(const Tensor& B) {
        return TensorLinearOpsGpu::addInplaceGpu<Tensor>(*this, B);
    }

    Tensor subtractCpu(const Tensor& B) const {
        assert(cols == B.cols && rows == B.rows);
        
        return TensorLinearOpsCpu::subtractCpu<Tensor>(*this, B);
    }

    Tensor subtractGpu(const Tensor& B) const {
        return TensorLinearOpsGpu::subtractGpu<Tensor>(*this, B);
    }
    
    Tensor& subtractInplaceCpu(const Tensor& B) {
        assert(rows == B.rows && cols == B.cols);

        return TensorLinearOpsCpu::subtractInplaceCpu<Tensor>(*this, B);
    }

    Tensor& subtractInplaceGpu(const Tensor& B) {
        return TensorLinearOpsGpu::subtractInplaceGpu<Tensor>(*this, B);
    }

    Tensor scalarScaleCpu(const computeT scalar) const {
        return TensorLinearOpsCpu::scalarScaleCpu<Tensor>(*this, scalar);
    }

    Tensor scalarScaleGpu(const computeT scalar) const {
        return TensorLinearOpsGpu::scalarScaleGpu<Tensor>(*this, scalar);
    }
     
    Tensor& scalarScaleInplaceCpu(const computeT scalar) {
        return TensorLinearOpsCpu::scalarScaleInplaceCpu<Tensor>(*this, scalar);
    }

    Tensor& scalarScaleInplaceGpu(const computeT scalar) {
        return TensorLinearOpsGpu::scalarScaleInplaceGpu<Tensor>(*this, scalar);
    }

    Tensor scalarDivideCpu(const computeT scalar) const {
        return TensorLinearOpsCpu::scalarDivideCpu<Tensor>(*this, scalar);
    }

    Tensor scalarDivideGpu(const computeT scalar) const {
        return TensorLinearOpsGpu::scalarDivideGpu<Tensor>(*this, scalar);
    }
    
    Tensor& scalarDivideInplaceCpu(const computeT scalar) {
        return TensorLinearOpsCpu::scalarDivideInplaceCpu<Tensor>(*this, scalar);
    }

    Tensor& scalarDivideInplaceGpu(const computeT scalar) {
        return TensorLinearOpsGpu::scalarDivideInplaceGpu<Tensor>(*this, scalar);
    }

    Tensor scalarAddCpu(const computeT scalar) const {
        return TensorLinearOpsCpu::scalarAddCpu<Tensor>(*this, scalar);
    }

    Tensor scalarAddGpu(const computeT scalar) const {
        return TensorLinearOpsGpu::scalarAddGpu<Tensor>(*this, scalar);
    }
       
    Tensor& scalarAddInplaceCpu(const computeT scalar) {
        return TensorLinearOpsCpu::scalarAddInplaceCpu<Tensor>(*this, scalar);
    }

    Tensor& scalarAddInplaceGpu(const computeT scalar) {
        return TensorLinearOpsGpu::scalarAddInplaceGpu<Tensor>(*this, scalar);
    }

    Tensor scalarSubtractCpu(const computeT scalar) const {
        return TensorLinearOpsCpu::scalarSubtractCpu<Tensor>(*this, scalar);
    }

    Tensor scalarSubtractGpu(const computeT scalar) const {
        return TensorLinearOpsGpu::scalarSubtractGpu<Tensor>(*this, scalar);
    }

    Tensor& scalarSubtractInplaceCpu(const computeT scalar) {
        return TensorLinearOpsCpu::scalarSubtractInplaceCpu<Tensor>(*this, scalar);
    }

    Tensor& scalarSubtractInplaceGpu(const computeT scalar) {
        return TensorLinearOpsGpu::scalarSubtractInplaceGpu<Tensor>(*this, scalar);
    }

    Tensor squareCpu() const {
        return TensorLinearOpsCpu::squareCpu<Tensor>(*this);
    }

    Tensor squareGpu() const {
        return TensorLinearOpsGpu::squareGpu<Tensor>(*this);
    }
      
    Tensor& squareInplaceCpu() {
        return TensorLinearOpsCpu::squareInplaceCpu<Tensor>(*this);
    }

    Tensor& squareInplaceGpu() {
        return TensorLinearOpsGpu::squareInplaceGpu<Tensor>(*this);
    }

    Tensor sqrtCpu() const {
        return TensorLinearOpsCpu::sqrtCpu<Tensor>(*this);
    }

    Tensor sqrtGpu() const {
        return TensorLinearOpsGpu::sqrtGpu<Tensor>(*this);
    }
     
    Tensor& sqrtInplaceCpu() {
        return TensorLinearOpsCpu::sqrtInplaceCpu<Tensor>(*this);
    }

    Tensor& sqrtInplaceGpu() {
        return TensorLinearOpsGpu::sqrtInplaceGpu<Tensor>(*this);
    }
    
    Tensor addBroadcastRowCpu(const Tensor& B) const {
        assert(cols == B.cols);

        Tensor C(rows, cols);

        const storageT* a = cpuData();
        const storageT* b = B.cpuData();
        storageT* c = C.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                computeT aVal = toCompute(a[offset(i, j)]);
                computeT bVal = toCompute(b[B.offset(0, j)]);

                computeT sum = aVal + bVal;

                c[C.offset(i, j)] = toStorage<storageT>(sum);
            }
        }

        return C;
    }

    Tensor addBroadcastRowGpu(Tensor& B) const;
     
    Tensor addBroadcastColCpu(const Tensor& B) const {
        assert(cols == B.cols);

        Tensor C(rows, cols);

        const storageT* a = cpuData();
        const storageT* b = B.cpuData();
        storageT* c = C.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                computeT aVal = toCompute(a[offset(i, j)]);
                computeT bVal = toCompute(b[B.offset(i, 0)]);
                
                computeT sum = aVal + bVal;

                c[C.offset(i, j)] = toStorage<storageT>(sum);
            }
        }

        return C;
    }

    Tensor addBroadcastColGpu(Tensor& B) const;
  
    Tensor multiplyBroadcastRowCpu(const Tensor& B) const {
        assert(cols == B.cols);

        Tensor C(rows, cols);

        const storageT* a = cpuData();
        const storageT* b = B.cpuData();
        storageT* c = C.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                computeT sum = computeT(0);

                computeT aVal = toCompute(a[offset(i, j)]);
                computeT bVal = toCompute(b[B.offset(0, j)]);
                
                sum = aVal * bVal;

                c[C.offset(i, j)] = toStorage<storageT>(sum);
            }
        }

        return C;
    }

    Tensor multiplyBroadcastRowGpu(Tensor& B) const;

    Tensor& multiplyBroadcastRowInplaceCpu(const Tensor& B) {
        assert(cols == B.cols);

        storageT* a = cpuData();
        const storageT* b = B.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                computeT aVal = toCompute(a[offset(i, j)]);
                computeT bVal = toCompute(b[B.offset(0, j)]);
                
                computeT sum = aVal * bVal;

                a[offset(i, j)] = toStorage<storageT>(sum);
            }
        }

        return *this;
    }

    Tensor& multiplyBroadcastRowInplaceGpu(const Tensor& B);
     
    Tensor multiplyBroadcastColCpu(const Tensor& B) const {
        assert(cols == B.cols);

        Tensor C(rows, cols);

        const storageT* a = cpuData();
        const storageT* b = B.cpuData();
        storageT* c = C.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                computeT aVal = toCompute(a[offset(i, j)]);
                computeT bVal = toCompute(b[B.offset(i, 0)]);
                
                computeT sum = aVal * bVal;

                c[C.offset(i, j)] = toStorage<storageT>(sum);
            }
        }

        return C;
    }

    Tensor multiplyBroadcastColGpu(Tensor& B) const;

    Tensor multiplyLastAxisMaskCpu(const Tensor& mask) const {
        return TensorLinearOpsCpu::multiplyLastAxisMaskCpu<Tensor>(*this, mask);
    }

    Tensor multiplyLastAxisMaskGpu(const Tensor& mask) const {
        return TensorLinearOpsGpu::multiplyLastAxisMaskGpu<Tensor>(*this, mask);
    }

    storageT sumCpu() const{
        return TensorLinearOpsCpu::sumCpu<Tensor>(*this);
    }

    storageT sumGpu() const {
        return TensorLinearOpsGpu::sumGpu<Tensor>(*this);
    }
 
    Tensor sumRowsCpu() const {
        Tensor result(1, cols);

        const storageT* a = cpuData();
        storageT* r = result.cpuData();

        for(size_t j = 0; j < cols; j++) {
            computeT sum = computeT(0);

            for(size_t i = 0; i < rows; i++) {
                computeT aVal = toCompute(a[offset(i, j)]);
                sum += aVal;
            }

            r[j] = toStorage<storageT>(sum);
        }

        return result;
    }

    Tensor sumRowsGpu() const;

    Tensor sumColsCpu() const{
        Tensor result(rows, 1);

        const storageT* a = cpuData();
        storageT* r = result.cpuData();

        for(size_t i = 0; i < rows; i++) {
            computeT sum = computeT(0);

            for(size_t j = 0; j < cols; j++) {
                computeT aVal = toCompute(a[offset(i, j)]);

                sum += aVal;
            }

            r[i] = toStorage<storageT>(sum);
        }

        return result;
    }

    Tensor sumColsGpu() const;

    storageT sumRowCpu(size_t row) const {
        assert(row < rows);
        computeT sum = computeT(0);

        const storageT* a = cpuData();

        for(size_t j = 0; j < cols; j++) {
            computeT aVal = toCompute(a[offset(row, j)]);

            sum += aVal;
        }

        return toStorage<storageT>(sum);
    }

    storageT sumColCpu(size_t col) const {
        assert(col < cols);
        computeT sum = computeT(0);

        const storageT* a = cpuData();

        for(size_t i = 0; i < rows; i++) {
            computeT aVal = toCompute(a[offset(i, col)]);
            sum += aVal;
        }

        return toStorage<storageT>(sum);
    }

    storageT meanCpu() const {
        return TensorLinearOpsCpu::meanCpu<Tensor>(*this);
    }

    storageT meanGpu() const {
        return TensorLinearOpsGpu::meanGpu<Tensor>(*this);
    }

    storageT minCpu() const {
        assert(rows > 0 && cols > 0);
        
        return TensorLinearOpsCpu::minCpu<Tensor>(*this);
    }

    storageT minGpu() const {
        return TensorLinearOpsGpu::minGpu<Tensor>(*this);
    }

    storageT maxCpu() const {
        assert(rows > 0 && cols > 0);
        
        return TensorLinearOpsCpu::maxCpu<Tensor>(*this);
    }

    storageT maxGpu() const {
        return TensorLinearOpsGpu::maxGpu<Tensor>(*this);
    }

    Tensor minRowsCpu() const {
        Tensor result(1, cols);

        const storageT* a = cpuData();
        storageT* r = result.cpuData();

        for(size_t j = 0; j < cols; j++) {
            computeT minVal = toCompute(a[offset(0, j)]);

            for(size_t i = 1; i < rows; i++) {
                computeT otherVal = toCompute(a[offset(i, j)]);
                if(otherVal < minVal) {
                    otherVal = minVal;
                }
            }

            r[j] = toStorage<storageT>(minVal);
        }

        return result;
    }

    Tensor minRowsGpu() const;

    Tensor maxRowsCpu() const {
        Tensor result(1, cols);

        const storageT* a = cpuData();
        storageT* r = result.cpuData();

        for(size_t j = 0; j < cols; j++) {
            computeT maxVal = toCompute(a[offset(0, j)]);

            for(size_t i = 1; i < rows; i++) {
                computeT otherVal = toCompute(a[offset(i, j)]);
                if(otherVal > maxVal) {
                    maxVal = otherVal;
                }
            }

            r[j] = toStorage<storageT>(maxVal);
        }

        return result;
    }

    Tensor maxRowsGpu() const;

    size_t argminCpu() const  {
        assert(rows > 0 && cols > 0);

        return TensorLinearOpsCpu::argMinCpu<Tensor>(*this);
    }

    size_t argminGpu() const {
        return TensorLinearOpsGpu::argminGpu<Tensor>(*this);
    }

    size_t argminRowCpu(size_t row) const {
        assert(row < rows);

        size_t target = row * cols;
        size_t end = target + cols;

        const storageT* a = cpuData();

        size_t idx = target;
        computeT minVal = computeT(a[target]);

        for(size_t i = target + 1; i < end; i++) {
            computeT aVal = toCompute(a[i]);
            if(aVal < minVal) {
                minVal = aVal;
                idx = i;
            }
        }

        return idx - target;
    }

    Tensor<size_t, 2> argminRowsCpu() const {
        Tensor<size_t, 2> result(1, cols);

        const storageT* a = cpuData();
        size_t* r = result.cpuData();

        for(size_t j = 0; j < cols; j++) {
            size_t idx = 0;
            computeT minVal = toCompute(a[offset(0, j)]);
            for(size_t i = 1; i < rows; i++) {
                computeT val = toCompute(a[offset(i, j)]);

                if(val < minVal) {
                    minVal = val;
                    idx = i;
                }

                r[j] = idx;
            }
        }

        return result;
    }


    Tensor<size_t, 2> argminRowsGpu() const;

    size_t argmaxCpu() const {
        assert(rows > 0 && cols > 0);

        return TensorLinearOpsCpu::argMaxCpu<Tensor>(*this);
    }

    size_t argmaxGpu() const {
        return TensorLinearOpsGpu::argmaxGpu<Tensor>(*this);
    }

    size_t argmaxRowCpu(size_t row) const {
        assert(row < rows);

        size_t target = row * cols;
        size_t end = target + cols;

        const storageT* a = cpuData();

        size_t idx = target;
        computeT maxVal = toCompute(a[target]);

        for(size_t i = target + 1; i < end; i++) {
            computeT otherVal = toCompute(a[i]);
            if(otherVal > maxVal) {
                maxVal = otherVal;
                idx = i;
            }
        }

        return idx - target;
    }

    Tensor<size_t, 2> argmaxRowsCpu() const {
        Tensor<size_t, 2> result(1, cols);

        const storageT* a = cpuData();
        size_t* r = result.cpuData();

        for(size_t j = 0; j < cols; j++) {
            size_t idx = 0;
            computeT maxVal = toCompute(a[offset(0, j)]);

            for(size_t i = 1; i < rows; i++) {
                computeT val = toCompute(a[offset(i, j)]);

                if(val > maxVal) {
                    maxVal = val;
                    idx = i;
                }

                r[j] = idx;
            }
        }

        return result;
    }

    Tensor<size_t, 2> argmaxRowsGpu() const;

    storageT normCpu() const {
        return TensorLinearOpsCpu::normCpu<Tensor>(*this);
    }

    storageT normGpu() const {
        return TensorLinearOpsGpu::normGpu<Tensor>(*this);
    }
    
    Tensor& normalizeCpu() {
        return TensorLinearOpsCpu::normalizeCpu<Tensor>(*this);
    }

    Tensor& normalizeGpu() {
        return TensorLinearOpsGpu::normalizeGpu<Tensor>(*this);
    }

    Tensor transposeCpu() const {
        Tensor B(cols, rows);

        const storageT* a = cpuData();
        storageT* b = B.cpuData();

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                b[B.offset(j, i)] = a[offset(i, j)];
            }
        }

        return B;
    }
    
    Tensor transposeGpu() const;

    Tensor& fillZeroCpu() {
        return TensorLinearOpsCpu::fillZeroCpu<Tensor>(*this);
    }

    Tensor& fillZeroGpu() {
        return TensorLinearOpsGpu::fillZeroGpu<Tensor>(*this);
    }

    Tensor& fillOneCpu() {
        return TensorLinearOpsCpu::fillOneCpu<Tensor>(*this);
    }

    Tensor& fillOneGpu() {
        return TensorLinearOpsGpu::fillZeroGpu<Tensor>(*this);
    }

    //Sqezzed into captivity
    Tensor& clipCpu(computeT minVal, computeT maxVal) {
        return TensorLinearOpsCpu::clipCpu<Tensor>(*this, minVal, maxVal);
    }

    Tensor& clipGpu(computeT minVal, computeT maxVal) {
        return TensorLinearOpsGpu::clipGpu<Tensor>(*this, minVal, maxVal);
    }
 
    Tensor& reshapeCpu(size_t newrows, size_t newcols) {
        assert(newrows * newcols == rows * cols);

        rows = newrows;
        cols = newcols;

        return *this;
    }

    Tensor& flattenCpu() {
        return reshapeCpu(1, rows * cols);
    }

    Tensor& initXavierCpu() {
        size_t fan = cols + rows;

        return TensorLinearOpsCpu::initXavierCpu<Tensor>(*this, fan);
    }

    Tensor& initHeCpu() {
       size_t fanIn = cols;

        return TensorLinearOpsCpu::initHeCpu<Tensor>(*this, fanIn);
    }
    
    bool equalsCpu(const Tensor& B) const {
        if(rows != B.rows || cols != B.cols) {
            return false;
        }

        return TensorLinearOpsCpu::equalsCpu<Tensor>(*this, B);
    }
};

}   //namespace alya
