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

add_executable(lu_dense_cpu lu_dense_cpu.cpp)
target_link_libraries(lu_dense_cpu Eigen3::Eigen)
add_executable(lu_dense_gpu lu_dense_gpu.cu)
add_executable(lu_dense_gpu_fancy lu_dense_gpu_fancy.cu)
add_executable(qr_dense_cpu qr_dense_cpu.cpp)
target_link_libraries(qr_dense_cpu Eigen3::Eigen)
