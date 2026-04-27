cmake_minimum_required(VERSION 3.27)

set(CMAKE_CUDA_COMPILER "nvcc")

if(WIN32)
    set(CMAKE_CXX_COMPILER "cl.exe")
    set(CMAKE_C_COMPILER "cl.exe")
endif()

project(AI LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()


foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(ARCH MATCHES "^[0-9]+$" AND ARCH LESS 80)
        message(WARNING
            "CUDA architecture sm_${ARCH} detected. "
            "Alya requires sm_80+ for native fp16/bf16 support (Ampere or newer). "
            "Older GPUs may not work correctly."
        )
    endif()
endforeach()

file(GLOB_RECURSE CUDA_SOURCES "cudaKernel/*.cu")

set(SOURCES
    src/main.cpp
    ${CUDA_SOURCES}
)

add_executable(Ai ${SOURCES})

set_target_properties(Ai PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

target_include_directories(Ai PRIVATE
    include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(Ai PRIVATE
    CUDA::cudart
)

target_compile_options(Ai PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O2>>
)