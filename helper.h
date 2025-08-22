#include <cassert>
#include <cmath>
#include <iostream>
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
