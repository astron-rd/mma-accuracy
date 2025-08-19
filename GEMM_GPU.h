#include <cuda_runtime.h>

#include "GEMM.h"

class GEMM_GPU : public GEMM {
public:
  GEMM_GPU(size_t M, size_t N, size_t K);
  ~GEMM_GPU();
  void compute(const float *A, const float *B, float *C) override;

private:
  virtual void launch(const float *A, const float *B, float *C) = 0;

  float *d_a_;
  float *d_b_;
  float *d_c_;
};
