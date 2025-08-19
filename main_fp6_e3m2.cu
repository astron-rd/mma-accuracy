#include "GEMM.h"
#include "GEMM_E3M2.h"
#include "helper.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 8;
  constexpr size_t K = 32;

  GEMM reference(M, N, K);
  GEMM_E3M2 gpu(M, N, K);

  const double min_normal_value = FP6_E3M2_MIN_NORMAL;
  const double max_normal_value = FP6_E3M2_MAX_NORMAL;

  run_test(M, N, K, reference, gpu, min_normal_value, max_normal_value);

  return 0;
}
