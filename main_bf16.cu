#include "GEMM.h"
#include "GEMM_BF16.h"
#include "helper.h"

int main() {
  constexpr size_t M = 16;
  constexpr size_t N = 16;
  constexpr size_t K = 16;

  GEMM reference(M, N, K);
  GEMM_BF16 gpu(M, N, K);

  run_test(M, N, K, reference, gpu, BF16_MIN_NORMAL, BF16_MAX_NORMAL);

  return 0;
}
