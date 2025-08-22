#include <gemm.h>

#include "test_range.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 16;
  constexpr size_t K = 16;

  GEMM_FP32 gpu(M, N, K);

  run_test(M, N, K, gpu);

  return 0;
}
