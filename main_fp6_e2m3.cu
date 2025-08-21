#include "GEMM.h"
#include "GEMM_E2M3.h"
#include "helper.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 8;
  constexpr size_t K = 32;

  GEMM reference(M, N, K);
  GEMM_E2M3 gpu(M, N, K);

  run_test(M, N, K, reference, gpu, FP6_E2M3_MIN_NORMAL, FP6_E2M3_MAX_NORMAL);

  return 0;
}
