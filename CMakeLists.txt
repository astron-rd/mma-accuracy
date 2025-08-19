cmake_minimum_required(VERSION 3.17)

project(mma-accuracy CXX CUDA)

enable_language(CUDA)

add_executable(test-fp32 main_fp32.cu GEMM.cpp GEMM_GPU.cu kernel_fp32.cu)
add_executable(test-fp16 main_fp16.cu GEMM.cpp GEMM_GPU.cu kernel_fp16.cu)
add_executable(test-bf16 main_fp16.cu GEMM.cpp GEMM_GPU.cu kernel_bf16.cu)
add_executable(test-fp8-e4m3 main_fp8_e4m3.cu GEMM.cpp GEMM_GPU.cu
                             kernel_fp8_e4m3.cu)
add_executable(test-fp8-e5m2 main_fp8_e5m2.cu GEMM.cpp GEMM_GPU.cu
                             kernel_fp8_e5m2.cu)
add_executable(test-fp6-e2m3 main_fp6_e2m3.cu GEMM.cpp GEMM_GPU.cu
                             kernel_fp6_e2m3.cu)
add_executable(test-fp6-e3m2 main_fp6_e3m2.cu GEMM.cpp GEMM_GPU.cu
                             kernel_fp6_e3m2.cu)
