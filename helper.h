#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "GEMM.h"

double inline compute_max_normal_value(int exponent_bits, int mantissa_bits) {
  const double bias = std::pow(2, exponent_bits - 1) - 1;
  const double max_exponent =
      std::pow(2, exponent_bits) - 2 - bias; // -2 because all 1s is special
  const double max_mantissa = 2.0 - std::pow(2, -mantissa_bits);
  return max_mantissa * std::pow(2, max_exponent);
}

double inline compute_max_normal_value_nvidia(int exponent_bits,
                                              int mantissa_bits) {
  const double bias = std::pow(2, exponent_bits - 1) - 1;
  const double max_exponent =
      std::pow(2, exponent_bits) - 2 - bias; // -2 because all 1s is special
  const double max_mantissa = 2.0 - std::pow(2, -mantissa_bits);
  return max_mantissa *
         std::pow(2, max_exponent + 1); // +1 for NVIDIA's double exponent
}

double inline compute_min_normal_value(int exponent_bits, int mantissa_bits) {
  const double bias = std::pow(2, exponent_bits - 1) - 1;
  return std::pow(2, 1 - bias);
}

#define FP32_MAX_NORMAL compute_max_normal_value(8, 23)
#define FP32_MIN_NORMAL compute_min_normal_value(8, 23)

#define BF16_MAX_NORMAL compute_max_normal_value(8, 7)
#define BF16_MIN_NORMAL compute_min_normal_value(8, 7)

#define FP16_MAX_NORMAL compute_max_normal_value(5, 10)
#define FP16_MIN_NORMAL compute_min_normal_value(5, 10)

#define FP8_E4M3_MAX_NORMAL compute_max_normal_value_nvidia(4, 3)
#define FP8_E4M3_MIN_NORMAL compute_min_normal_value(4, 3)

#define FP8_E5M2_MAX_NORMAL compute_max_normal_value(5, 2)
#define FP8_E5M2_MIN_NORMAL compute_min_normal_value(5, 2)

#define FP6_E2M3_MAX_NORMAL compute_max_normal_value(2, 3)
#define FP6_E2M3_MIN_NORMAL compute_min_normal_value(2, 3)

#define FP6_E3M2_MAX_NORMAL compute_max_normal_value(3, 2)
#define FP6_E3M2_MIN_NORMAL compute_min_normal_value(3, 2)

template <typename T>
std::pair<T, T> compare_vectors(const std::vector<T> &a,
                                const std::vector<T> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vectors must have the same size.");
  }

  double abs_sum = 0.0;
  double ref_sum = 0.0;

  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i])) {
      continue; // Ignore NaN values
    }
    if (std::isinf(a[i]) || std::isinf(b[i])) {
      continue; // Ignore Inf values
    }

    double diff =
        static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
    abs_sum += diff * diff;
    ref_sum += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
  }

  double abs_error = std::sqrt(abs_sum);
  double rel_error = (ref_sum > 1e-6) ? abs_error / std::sqrt(ref_sum) : 0.0L;

  return {abs_error, rel_error};
}

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