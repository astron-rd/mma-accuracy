#include "GEMM.h"
#include "GEMM_FP16.h"
#include "helper.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 16;
  constexpr size_t K = 16;

  GEMM reference(M, N, K);
  GEMM_FP16 gpu(M, N, K);

  const double min_normal_value = FP16_MIN_NORMAL;
  const double max_normal_value = FP16_MAX_NORMAL;

  run_test(M, N, K, reference, gpu, min_normal_value, max_normal_value);

  return 0;
}
