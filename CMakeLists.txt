cmake_minimum_required(VERSION 3.17)

project(mma-accuracy CXX CUDA)

enable_language(CUDA)

set(PRECISISONS fp32 fp16 bf16 fp8_e4m3 fp8_e5m2)

if(CMAKE_CUDA_ARCHITECTURE STREQUAL 120)
  list(APPEND precisions fp6_e2m3)
endif()

add_library(GEMM OBJECT GEMM.cpp GEMM_GPU.cu)
foreach(precision ${PRECISISONS})
  add_executable(test-${precision} main_${precision}.cu kernel_${precision}.cu)
  target_link_libraries(test-${precision} $<TARGET_OBJECTS:GEMM>)
endforeach()
