#define __HIP_PLATFORM_AMD__
#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <hip/hip_runtime.h>
#define GAP 4
#define NUM_BINS (26 + GAP - 1) / GAP
#define BLOCK_SIZE 1024
#define COARS_FACT 400

// Basic histogram kernel
__global__ void histogram_kernel(char *input, unsigned int *output, unsigned int n);
void histogram(char *input, unsigned int *output, unsigned int n);

// Histogram kernel with privatization using RAM
__global__ void histogram_private_kernel(char *input, unsigned int *output, unsigned int n);
void histogram_private(char *input, unsigned int *output, unsigned int n);

// Histogram kernel with privatization using shared memory
__global__ void histo_private_shared_kernel(char *data, unsigned int *histo, unsigned int len);
void histo_private_shared(char *data, unsigned int *histo, unsigned int len);

// Histogram kernel with coarsening using contiguous partitioning
__global__ void histo_tc_cont_kernel(char *data, unsigned int *histo, unsigned int len);
void histo_tc_cont(char *data, unsigned int *histo, unsigned int len);

// Histogram kernel with coarsening using interleaved partitioning
__global__ void histo_tc_interl_kernel(char *data, unsigned int *histo, unsigned int len);
void histo_tc_interl(char *data, unsigned int *histo, unsigned int len);

#endif