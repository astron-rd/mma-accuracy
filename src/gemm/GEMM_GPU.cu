#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "gemm/gpu/GEMM_GPU.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

GEMM_GPU::GEMM_GPU(size_t M, size_t N, size_t K) : GEMM(M, N, K) {
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  std::cout << "GPU: " << deviceProp.name << " (SM " << deviceProp.major << "."
            << deviceProp.minor << ")" << std::endl;

  CUDA_CHECK(cudaMalloc(&d_a_, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&d_b_, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc(&d_c_, sizeof(float) * M * N));
}

GEMM_GPU::~GEMM_GPU() {
  CUDA_CHECK(cudaFree(d_a_));
  CUDA_CHECK(cudaFree(d_b_));
  CUDA_CHECK(cudaFree(d_c_));
}

__global__ void kernel(const float *A, const float *B, float *C);

void GEMM_GPU::compute(const float *A, const float *B, float *C) {
  CUDA_CHECK(
      cudaMemcpy(d_a_, A, sizeof(float) * M_ * K_, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b_, B, sizeof(float) * K_ * N_, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_c_, C, sizeof(float) * M_ * N_, cudaMemcpyHostToDevice));

  dim3 threads(32);
  dim3 grid(1);

  kernel<<<grid, threads>>>(d_a_, d_b_, d_c_);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(
      cudaMemcpy(C, d_c_, sizeof(float) * M_ * N_, cudaMemcpyDeviceToHost));
}
