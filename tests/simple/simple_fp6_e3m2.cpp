#include <gemm.h>

#include "test_simple.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 8;
  constexpr size_t K = 32;

  GEMM reference(M, N, K);
  GEMM_E3M2 gpu(M, N, K);

  run_test(M, N, K, reference, gpu);

  return 0;
}
