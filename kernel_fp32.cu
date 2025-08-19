constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

__global__ void kernel(const float *A, const float *B, float *C) {
  for (int m = threadIdx.x; m < M; m += blockDim.x) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        const float a = A[m * K + k]; // A is row-major (MxK)
        const float b = B[n * K + k]; // B is column-major (KxN)
        sum += a * b;
      }
      C[m * N + n] = sum; // Row-major output
    }
  }
}
