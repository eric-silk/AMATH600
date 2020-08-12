# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
cmake_minimum_required (VERSION 3.0)
project(LU_GPU LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test_sparse_matrix test_sparse_matrix.cu)
