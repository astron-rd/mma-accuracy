#include "GEMM.h"

GEMM::GEMM(size_t M, size_t N, size_t K) : M_(M), N_(N), K_(K) {}

void GEMM::compute(const float *A, const float *B, float *C) {
  for (int m = 0; m < M_; m++) {
    for (int n = 0; n < N_; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K_; k++) {
        const float a = A[m * K_ + k]; // A is row-major (MxK)
        const float b = B[n * K_ + k]; // B is column-major (KxN)
        sum += a * b;
      }
      C[m * N_ + n] = sum; // Row-major output
    }
  }
}
