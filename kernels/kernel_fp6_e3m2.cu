#include <cuda_fp6.h>
#include <mma.h>

using namespace nvcuda::wmma;

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 32;

using DTYPE = __nv_fp6_e3m2;

template <>
class fragment<matrix_a, M, N, K, DTYPE, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, M, N, K, DTYPE, col_major>
    : public __frag_base<int, 2> {};

template <>
class fragment<accumulator, M, N, K, float> : public __frag_base<float, 4> {};

inline __device__ void
mma_sync_ptx(fragment<accumulator, M, N, K, float> &d,
             const fragment<matrix_a, M, N, K, DTYPE, row_major> &a,
             const fragment<matrix_b, M, N, K, DTYPE, col_major> &b,
             const fragment<accumulator, M, N, K, float> &c) {
  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32 {%0, "
      "%1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}

__device__ inline void
store_matrix_sync(float *p, const fragment<accumulator, M, N, K, float> &d,
                  unsigned ldm, nvcuda::wmma::layout_t layout) {
  ((float2 *)p)[ldm / 2 * (threadIdx.x / 4) + threadIdx.x % 4] =
      make_float2(d.x[0], d.x[1]);
  ((float2 *)p)[ldm / 2 * (threadIdx.x / 4 + 8) + threadIdx.x % 4] =
      make_float2(d.x[2], d.x[3]);
}

inline __device__ unsigned laneid() { return threadIdx.x; }

template <typename T>
inline __device__ void
load_matrix_sync(fragment<matrix_a, M, N, K, T, row_major> &a, const void *p,
                 unsigned ldm) {
  a.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  a.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4];
  a.x[2] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
  a.x[3] = ((const int *)p)[ldm / 4 * (laneid() / 4 + 8) + laneid() % 4 + 4];
}

template <typename T>
inline __device__ void
load_matrix_sync(fragment<matrix_b, M, N, K, T, col_major> &b, const void *p,
                 unsigned ldm) {
  b.x[0] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4];
  b.x[1] = ((const int *)p)[ldm / 4 * (laneid() / 4) + laneid() % 4 + 4];
}

__global__ void kernel(const float *A, const float *B, float *C) {
  if (threadIdx.x >= 32) {
    return;
  }

  fragment<matrix_a, M, N, K, DTYPE, row_major> a;
  fragment<matrix_b, M, N, K, DTYPE, col_major> b;
  fragment<accumulator, M, N, K, float> c, d;
  fill_fragment(c, 0.0f);
  fill_fragment(d, 0.0f);

  // Convert FP32 inputs to FP6 storage format
  __shared__ __nv_fp6_storage_t tmpA[M * K];
  __shared__ __nv_fp6_storage_t tmpB[K * N];

  for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
    tmpA[i] = __nv_cvt_float_to_fp6(A[i], __NV_E3M2, cudaRoundNearest);
  }

  for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
    tmpB[i] = __nv_cvt_float_to_fp6(B[i], __NV_E3M2, cudaRoundNearest);
  }

  __syncthreads();

  // Load fragments from the FP6 arrays
  load_matrix_sync(a, reinterpret_cast<const DTYPE *>(tmpA), K);
  load_matrix_sync(b, reinterpret_cast<const DTYPE *>(tmpB), K);

  // Perform matrix multiplication
  mma_sync_ptx(c, a, b, d);

  // Store result
  store_matrix_sync(C, c, N, nvcuda::wmma::mem_row_major);
}