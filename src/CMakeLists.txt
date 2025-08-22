add_library(GEMM OBJECT gemm/GEMM.cpp gemm/GEMM_GPU.cu)
target_include_directories(GEMM PUBLIC ${CMAKE_SOURCE_DIR}/include)
