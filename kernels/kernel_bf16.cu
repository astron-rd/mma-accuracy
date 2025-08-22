#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

using DTYPE = __nv_bfloat16;

__global__ void kernel(const float *A, const float *B, float *C) {
  if (threadIdx.x >= 32) {
    return;
  }

  fragment<matrix_a, M, N, K, DTYPE, row_major> a;
  fragment<matrix_b, M, N, K, DTYPE, col_major> b;
  fragment<accumulator, M, N, K, float> c;

  __shared__ DTYPE tmpA[M * K];
  __shared__ DTYPE tmpB[K * N];

  for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
    tmpA[i] = __float2bfloat16(A[i]);
  }

  for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
    tmpB[i] = __float2bfloat16(B[i]);
  }

  __syncthreads();

  // Load fragments
  load_matrix_sync(a, tmpA, K);
  load_matrix_sync(b, tmpB, K);
  load_matrix_sync(c, C, N, mem_row_major);

  // Perform matrix multiplication
  mma_sync(c, a, b, c);

  // Store result
  store_matrix_sync(C, c, N, nvcuda::wmma::mem_row_major);
}
