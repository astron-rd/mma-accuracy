cmake_minimum_required(VERSION 3.17)

project(mma-accuracy CXX CUDA)
enable_language(CUDA)

# Determine which data types are available based on the CUDA architectures FP32
# and FP16 are always available
set(HAVE_FP32 ON)
set(HAVE_FP16 ON)

foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  # Strip non-numeric suffixes (e.g. "90a" -> "90")
  string(REGEX MATCH "^[0-9]+" arch_num "${arch}")

  # BF16 is available from 80 onward
  if(arch_num GREATER_EQUAL 80)
    set(HAVE_BF16 ON)
  endif()

  # FP8 is available from 89 onward
  if(arch_num GREATER_EQUAL 89)
    set(HAVE_FP8 ON)
  endif()

  # FP6 is available from 120 onward
  if(arch_num GREATER_EQUAL 120)
    set(HAVE_FP6 ON)
  endif()
endforeach()

# Generate gemm_config.h based on available data types
configure_file(cmake/gemm_config.h.in
               "${CMAKE_BINARY_DIR}/include/gemm_config.h")

# Add include directory for generated gemm_config.h
include_directories("${CMAKE_BINARY_DIR}/include")

# Add main include directory
include_directories("${CMAKE_SOURCE_DIR}/include")

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
