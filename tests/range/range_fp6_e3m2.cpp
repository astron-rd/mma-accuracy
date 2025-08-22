#include <gemm.h>

#include "test_range.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 8;
  constexpr size_t K = 32;

  GEMM_E3M2 gpu(M, N, K);

  run_test(M, N, K, gpu);

  return 0;
}
