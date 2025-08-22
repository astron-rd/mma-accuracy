#include <gemm_config.h>

#include "gemm/GEMM.h"
#ifdef HAVE_FP16
#include "gemm/gpu/GEMM_FP32.cuh"
#endif
#ifdef HAVE_FP16
#include "gemm/gpu/GEMM_FP16.cuh"
#endif
#ifdef HAVE_BF16
#include "gemm/gpu/GEMM_BF16.cuh"
#endif
#ifdef HAVE_FP8
#include "gemm/gpu/GEMM_E4M3.cuh"
#include "gemm/gpu/GEMM_E5M2.cuh"
#endif
#ifdef HAVE_FP6
#include "gemm/gpu/GEMM_E2M3.cuh"
#include "gemm/gpu/GEMM_E3M2.cuh"
#endif