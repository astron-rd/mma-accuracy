#pragma once

#include "GEMM_GPU.h"

__global__ void kernel(const float *A, const float *B, float *C);
class GEMM_E2M3 : public GEMM_GPU {
public:
  GEMM_E2M3(size_t M, size_t N, size_t K) : GEMM_GPU(M, N, K){};

private:
  void launch(const float *d_a, const float *d_b, float *d_c) override {
    dim3 threads(32);
    dim3 grid(1);
    kernel<<<grid, threads>>>(d_a, d_b, d_c);
  }
};