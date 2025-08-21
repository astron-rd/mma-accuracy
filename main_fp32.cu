#include "GEMM.h"
#include "GEMM_FP32.h"
#include "helper.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 16;
  constexpr size_t K = 16;

  GEMM reference(M, N, K);
  GEMM_FP32 gpu(M, N, K);

  run_test(M, N, K, reference, gpu, FP32_MIN_NORMAL, FP32_MAX_NORMAL);

  return 0;
}
