#pragma once

#include <cstdlib>

class GEMM {
public:
  GEMM(size_t M, size_t N, size_t K);
  virtual ~GEMM() {}
  virtual void compute(const float *A, const float *B, float *C);

protected:
  size_t M_, N_, K_;
};