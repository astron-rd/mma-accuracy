#include <algorithm>
#include <random>
#include <vector>

#include <gemm.h>

#include "helper.h"

void run_test(int M, int N, int K, GEMM &gemm) {
  std::vector<float> a(M * K, 1.0f);
  std::vector<float> b(K * N, 1.0f / K);
  std::vector<float> c(M * N);

  // Workaround for (1.0f / 32) not representable in FP6
  if (M == 16 && N == 8 && K == 32) {
    std::fill(a.begin(), a.end(), 0.25f);
    std::fill(b.begin(), b.end(), 0.125f);
  }
  auto test_value = [&](float val) {
    std::fill(c.begin(), c.end(), 0.0f);
    c[0] = val;
    gemm.compute(a.data(), b.data(), c.data());
    return c[0] == val + 1.0f;
  };

  // Phase 1: exponential search to find an upper bound
  float low = 0.0f;
  float high = 1.0f;
  while (high <= FP32_MAX_NORMAL && test_value(high)) {
    std::cout << high << " + 1 = " << c[0] << " (ok)" << std::endl;
    low = high;
    high *= 2.0f;
  }
  const float diff = high + 1 - c[0];
  if (high <= FP32_MAX_NORMAL) {
    std::cout << std::fixed << static_cast<size_t>(high)
              << " + 1 != " << static_cast<size_t>(c[0]) << ", diff = " << diff
              << " (fail)" << std::endl;
  }

  // If we went past the largest FP32 normal value, everything is safe
  if (high > FP32_MAX_NORMAL) {
    std::cout << "No overflow" << std::endl;
    return;
  }

  // Phase 2: binary search between low and high
  while (low + 1 < high) {
    float mid = std::floor((low + high) / 2.0f);
    if (test_value(mid)) {
      low = mid; // still safe
    } else {
      high = mid; // overflow occurred
    }
  }

  std::cout << "Largest safe value: " << std::fixed << static_cast<size_t>(low)
            << std::endl;
  std::cout << "First failing value: " << std::fixed
            << static_cast<size_t>(high) << std::endl;
}