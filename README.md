# Alya
A hand-built C++/CUDA neural network framework

___

Alya is a neural network framework written from scratch in C++ and CUDA. It was built out of curiosity - to understand what actually happens inside, at implementation level. No PyTorch, no TensorFlow, no cuBLAS, no CUTLASS. Kernels are handwritten. Memory management is manual. Everything is implemented directly.

It was also my First C++ project. I learned the language while buiding this.

The architecture is **declarative**: you describe what you want, Alya builds it. Complexity is pushed inward - layers internals, memory management, and CUDA kernels stay hidden. But whether the model actually trains well is on you. Bad learning rate -> bad model.

The design is **not modeled after existing frameworks**. It reflects how i thought about the problem while learing - unconventional in places, but readable by necessity... i think.

## What can it do

- Fully connected layers (FC) with policy-based activation ops
- Dropout regularization
- MLP composition with layer, optimizer, trainer/validator seperation
- A few different losses
- Mixed precision: bf16/fp16/fp32/fp64
- AdamW and SGD with weight decay as optimizers
- IDX dataset loading (MNIST, EMNIST variants) and single file csv with "," separation
- Integrated validator

## Usage

Define your architecture, attach a loss and optimizer, hand it over to the trainer:

```cpp
using P = alya::bf16;

alya::Device gpu;
gpu.type = alya::DeviceType::GPU;

Hyperparameter
//...
Loading Data
//...

alya::FC<P, LeakyReLuOp> layer1(784, 512);
alya::Dropout<P> drop1(layer1, 0.03f);
alya::FC<P, LeakyReLuOp> layer2(512, 384);
//...

alya::MLP<P> model;
model.addLayer(&layer1);
model.addLayer(&layer2);
model.addDropout(&drop1);
//...

alya::CrossEntropyLoss<P> loss;
alya::AdamW<P> opt(LR, WD);

alya::Trainer<P> trainer(BATCH_SIZE, 784, 10, model, dataset, testDataset, loss, opt, EPOCHS, gpu);
trainer.train();
```
Full example in src/main.cpp.

Operation like matmul, sum etc. are accessible as well as Tensors but not required.

## Requirements

- NVIDIA GPU with ampere architecture or newer (RTX 30xx+) - older GPUs may work but fp16/bf16 operations are not natively supported and will be significantly slower or may not function correctly
- [CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-downloads)
- [CMake 3.27+](https://cmake.org/download/)
- **Windows:** Visual Studio 2022+ with "Desktop development with C++" workload
- **Linux:** GCC 10+ or Clang 10+

## Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

If CMake doesn't detect your GPU architecture correctly, specify it manually:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=your number
```

| Architecture | GPU |
|---|---|
| 86 | RTX 30xx (Ampere) |
| 89 | RTX 40xx (Ada) |
| 100 | RTX 50xx (Blackwell) |

**Note**: CMake may detects RTX 50xx as sm_75 (RTX 20xx). Manually setting is recommended

To compile on windows, developer command prompt for VS 20xx is often required.

## Datasets

The framework supports datasets in IDX format.

Download used datasets EMNIST from https://www.nist.gov/itl/products-and-services/emnist-dataset

You should have gzip.zip. You can use the python script to extract it at once. Make sure script and gzip.zip are in the same directory.

```bash
python extract_datasets.py
```

Requires Python 3. No additional dependencies.

## Tested Results

All runs: bf16, batch size 2048, 25 epochs, AdamW.

| Dataset | Classes | Train avg. acc | Val avg. acc | Train avg. loss |
| --- | --- | --- | --- | --- |
| MNIST | 10 | 99.16% | 98.25% | 0.0136 |
| EMNIST digits | 10 | 99.52% | 98.98% | 0.0007 |
| EMNIST letters | 26 | 94.36% | 90.99% | 0.0774 |
| EMNIST balanced | 47 | 87.77% | 83.58% | 0.1631 |
| EMNIST bymerge | 47 | 89.20% | 88.76% | 0.1468 |
| EMNIST byclass | 62 | 85.38% | 85% | 0.2016 |

## Tested Environment

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 5070 Ti (Blackwell, sm_100)
- **CUDA:** 13.2
- **Compiler:** MSVC 19.50 (Visual Studio 2026)
- **CMake:** 4.3

## Troubleshooting

- If CUDA Toolkit is not found, try reinstalling and choose manual/custom, not express and chosse integrate CUDA into VS environment.
- If CUDA can´t compile, check if environment Variable for CUDA is set.
- WSL must be WSL2 and CUDA drivers have to be WSL2 driver on windows, not linux cuda driver. See more here: https://developer.nvidia.com/cuda/wsl