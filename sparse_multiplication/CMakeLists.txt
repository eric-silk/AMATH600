# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
cmake_minimum_required (VERSION 3.0)
project(LU_GPU LANGUAGES CUDA CXX)

set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fno-elide-constructors" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -DNDEBUG"  CACHE STRING "" FORCE)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, defaulting to Release build")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_executable(test_sparse_matrix test_sparse_matrix.cu)
set_property(TARGET test_sparse_matrix PROPERTY CXX_STANDARD 14)
target_compile_options(test_sparse_matrix PRIVATE -g -G)

