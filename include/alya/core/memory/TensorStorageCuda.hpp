#pragma once

#include <alya/core/memory/Storage.hpp>

namespace alya::internal {

void tensorStorageAllocateGpu(Storage* storage);
void tensorStorageFreeGpu(Storage* storage) noexcept;
void tensorStorageSyncToGpu(Storage* storage);
void tensorStorageSyncToCpu(Storage* storage);

}