cmake_minimum_required(VERSION 3.18)
project(cudaHasher LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable position-independent code for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Java and JNI
find_package(Java REQUIRED)
find_package(JNI REQUIRED)

include_directories(${JNI_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add source files
add_library(cudaHasher SHARED
    cudaHasher.cpp
    sha256.cu
    md5.cu
    config.h
    sha256.cuh
    md5.cuh
)

# On Windows, CUDA toolkit path is often set in environment variable
if (WIN32)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(cudaHasher PRIVATE ${CUDAToolkit_LIBRARY_DIR}/cudart.lib)
endif()

# Use the runtime library directly if CUDAToolkit fails
target_link_libraries(cudaHasher PRIVATE cuda)
