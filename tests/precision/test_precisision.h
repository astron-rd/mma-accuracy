#include <algorithm>
#include <random>
#include <vector>

#include <gemm.h>

#include "helper.h"

template <typename T> std::vector<T> logspace(T start, T end, std::size_t n) {
  std::vector<T> out;
  out.reserve(n);

  if (n == 0)
    return out;
  if (n == 1) {
    out.push_back(start);
    return out;
  }

  if (!(start > T(0)) || !(end > T(0))) {
    throw std::invalid_argument("logspace requires positive start/end values.");
  }

  long double ls = std::log(static_cast<long double>(start));
  long double le = std::log(static_cast<long double>(end));
  long double step = (le - ls) / static_cast<long double>(n - 1);

  for (std::size_t i = 0; i < n; ++i) {
    long double li = ls + step * static_cast<long double>(i);
    long double vi = std::exp(li); // exp of the interpolated log
    out.push_back(static_cast<T>(vi));
  }
  return out;
}

void run_test(int M, int N, int K, GEMM &gemm1, GEMM &gemm2,
              double min_normal_value, double max_normal_value) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c1(M * N);
  std::vector<float> c2(M * N);

  std::mt19937 gen(0);

  std::pair<double, double> errors_sum(0.0, 0.0);

  for (double max_value : logspace(min_normal_value, max_normal_value, 32)) {
    std::uniform_real_distribution<float> dist_a(-max_value / 2, max_value / 2);
    std::normal_distribution<float> dist_b(1, 1e-1);

    const size_t nr_repetitions = 1;

    for (size_t i = 0; i < nr_repetitions; i++) {

      for (auto &x : a) {
        x = dist_a(gen);
      }
      for (auto &x : b) {
        x = dist_b(gen);
      }
      for (auto &x : c1) {
        x = 0.0f;
      }

      const float a_min = *std::min_element(a.begin(), a.end());
      const float a_max = *std::max_element(a.begin(), a.end());
      const float b_min = *std::min_element(b.begin(), b.end());
      const float b_max = *std::max_element(b.begin(), b.end());

      std::cout << "[ " << a_min << ", " << a_max << " ] "
                << "[ " << b_min << ", " << b_max << " ] ";

      gemm1.compute(a.data(), b.data(), c1.data());
      gemm2.compute(a.data(), b.data(), c2.data());

      auto errors = compare_vectors(c1, c2);
      errors_sum.first += errors.first;   // sum of relative errors
      errors_sum.second += errors.second; // sum of absolute errors
    }

    std::cout << " error: " << std::scientific
              << errors_sum.first / nr_repetitions << " "
              << errors_sum.second / nr_repetitions << std::endl;
  }
}