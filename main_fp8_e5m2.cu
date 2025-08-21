#include "GEMM.h"
#include "GEMM_E5M2.h"
#include "helper.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 8;
  constexpr size_t K = 32;

  GEMM reference(M, N, K);
  GEMM_E5M2 gpu(M, N, K);

  run_test(M, N, K, reference, gpu, FP8_E5M2_MIN_NORMAL, FP8_E5M2_MAX_NORMAL);

  return 0;
}
