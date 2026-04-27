#include <alya/layer/FC.hpp>
#include <alya/core/Precision/PrecisonTypes.cuh> 
#include <alya/layer/Dropout.hpp>
#include <alya/layer/MLP.hpp>
#include <alya/activation/ActivationOps.cuh>
#include <alya/losses/CeSo.hpp>
#include <alya/optimizer/AdamW.hpp>
#include <alya/core/data/Loader.hpp>
#include <alya/runtime/Trainer.hpp>
#include <alya/profiling/Timer.hpp>

int main() {
    alya::Timer timer;
    timer.start();

    using P = alya::bf16;
    
    alya::Device gpu;
    gpu.type = alya::DeviceType::GPU;
    gpu.gpu_id = 0;

    constexpr size_t BATCH_SIZE = 2048;
    constexpr float LR = 0.002f;
    constexpr float WD = 0.0003f;
    constexpr size_t EPOCHS = 25;
    
    alya::loader<P> emnist("datasets/byclass/emnist-byclass-train-images-idx3-ubyte", "datasets/byclass/emnist-byclass-train-labels-idx1-ubyte", BATCH_SIZE, alya::LoaderType::EMNISTBYCLASS);
    alya::loader<P> emnist_val("datasets/byclass/emnist-byclass-test-images-idx3-ubyte", "datasets/byclass/emnist-byclass-test-labels-idx1-ubyte", BATCH_SIZE, alya::LoaderType::EMNISTBYCLASS);
    emnist.normalize(alya::NormType::DIV255);
    emnist_val.normalize(alya::NormType::DIV255);
    emnist.normalize(alya::NormType::ZERO_MEAN_STD__GLOBAL);
    emnist_val.normalize(alya::NormType::ZERO_MEAN_STD__GLOBAL);

    alya::FC<P, LeakyReLuOp> layer1(784, 512);
    alya::Dropout<P> drop1(layer1, 0.03f);
    alya::FC<P, LeakyReLuOp> layer2(512, 384);
    alya::Dropout<P> drop2(layer2, 0.1f);
    alya::FC<P, LeakyReLuOp> layer3(384, 256);
    alya::Dropout<P> drop3(layer3, 0.125f);
    alya::FC<P, LeakyReLuOp> layer4(256, 128);
    alya::Dropout<P> drop4(layer4, 0.025f);
    alya::FC<P, LeakyReLuOp> layer5(128, 64);
    alya::FC<P, LinearOp> layer6(64, 62);

    alya::MLP<P> model;
    model.addLayer(&layer1);
    model.addLayer(&layer2);
    model.addLayer(&layer3);
    model.addLayer(&layer4);
    model.addLayer(&layer5);
    model.addLayer(&layer6);
    model.addDropout(&drop1);
    model.addDropout(&drop2);
    model.addDropout(&drop3);
    model.addDropout(&drop4);

    alya::CrossEntropyLoss<P> loss;
    alya::AdamW<P> opt(LR, WD);

    alya::Trainer<P> trainer(BATCH_SIZE, 784, 62, model, emnist, emnist_val, loss, opt, EPOCHS, gpu);
    trainer.train();

    timer.stop();
    timer.print<alya::time::seconds>();

    return 0;
}