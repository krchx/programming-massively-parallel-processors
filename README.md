# HIP/ROCm Kernel Implementations

## Contents

### Chapter 3

- **BlurImage**: Implements image blurring using HIP.
- **greyScaleConverter**: Converts images to grayscale using HIP.
- **matmul**: Matrix multiplication using HIP.

### Chapter 4

- **gpu_features.cpp**: Queries and displays GPU features.

### Chapter 5

- **gpu_features.cpp**: Queries and displays GPU features.
- **matmul_tiled_gen.cpp**: General matrix multiplication using tiling.
- **matmul_tiled_sq.cpp**: Square matrix multiplication using tiling.

### Chapter 7

- **conv_1.cpp**: 2D convolution using HIP.

### Chapter 8

- **stencil_kernel.hpp**: Header file for stencil kernels.
- **kernel_shared.cpp**: Shared memory stencil kernel.
- **kernel_tc.cpp**: Thread coarsening stencil kernel.

### Chapter 9

- **kernel.hpp**: Header file for histogram kernels.
- **kernel_basic.cpp**: Basic histogram kernel.
- **kernel_priv_shared.cpp**: Histogram kernel with privatization using shared memory.
- **kernel_private.cpp**: Histogram kernel with privatization using RAM.
- **kernel_tc_cont.cpp**: Histogram kernel with coarsening using contiguous partitioning.
- **kernel_tc_interleaved.cpp**: Histogram kernel with coarsening using interleaved partitioning.
- **kernel_aggregation.cpp**: Placeholder for histogram kernel with aggregation.
- **main.cpp**: Main file to test histogram kernels.

### Chapter 10

- **kernel.hpp**: Header file for sum reduction kernels.
- **kernel_arbitary_len.cpp**: Sum reduction kernel for arbitrary length arrays.
- **kernel_convergent.cpp**: Sum reduction kernel with less control divergence.
- **kernel_shared_mem.cpp**: Sum reduction kernel using shared memory.
- **kernel_simple.cpp**: Simple sum reduction kernel.
- **main.cpp**: Main file to test sum reduction kernels.

### Library

- **PngHandlib**: Library for handling PNG images.

### Prerequisites

- HIP/ROCm installed on your system.
- A compatible AMD GPU.
  (In the book, CUDA is used instead of HIP, so check the book if you have an NVIDIA GPU.)
