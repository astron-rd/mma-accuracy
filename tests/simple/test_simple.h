#include <algorithm>
#include <iomanip>
#include <random>
#include <vector>

#include <gemm.h>

#include "helper.h"

#include <cuda_fp8.h>

void print_matrix(const std::vector<float> &matrix, int rows, int cols,
                  const std::string &name = "Matrix") {
  std::cout << name << " (" << rows << "x" << cols << "):\n";

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      size_t index = i * cols + j;
      std::cout << std::setw(10) << std::setprecision(2) << matrix[index]
                << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

void run_test(int M, int N, int K, GEMM &gemm1, GEMM &gemm2) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c1(M * N);
  std::vector<float> c2(M * N);

  std::mt19937 gen(0);

  std::pair<double, double> errors_sum(0.0, 0.0);

  std::uniform_real_distribution<float> dist(-1, 1);

  for (auto &x : a) {
    x = dist(gen);
  }
  for (auto &x : b) {
    x = dist(gen);
  }
  for (auto &x : c1) {
    x = 0.0f;
  }

  const float a_min = *std::min_element(a.begin(), a.end());
  const float a_max = *std::max_element(a.begin(), a.end());
  const float b_min = *std::min_element(b.begin(), b.end());
  const float b_max = *std::max_element(b.begin(), b.end());

  std::cout << "range a: [" << a_min << ", " << a_max << " ]" << std::endl;
  std::cout << "range b: [" << b_min << ", " << b_max << " ]" << std::endl;

  gemm1.compute(a.data(), b.data(), c1.data());
  gemm2.compute(a.data(), b.data(), c2.data());

  const float c_min = *std::min_element(c1.begin(), c1.end());
  const float c_max = *std::max_element(c1.begin(), c1.end());
  std::cout << "range c: [" << c_min << ", " << c_max << " ]" << std::endl;

  auto errors = compare_vectors(c1, c2);
  errors_sum.first += errors.first;   // sum of relative errors
  errors_sum.second += errors.second; // sum of absolute errors

  std::cout << "error: " << std::scientific << errors_sum.first << " "
            << errors_sum.second << std::endl
            << std::endl;
  ;

  print_matrix(c1, M, N, "c1");
  print_matrix(c2, M, N, "c2");

  std::vector<float> diff(M * N);
  for (size_t i = 0; i < M * N; i++) {
    diff[i] = std::abs(c1[i] - c2[i]);
  }
  print_matrix(diff, M, N, "diff");
}