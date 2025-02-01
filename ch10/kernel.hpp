#ifndef KERNEL_HPP
#define KERNEL_HPP

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#define BLOCK_SIZE 1024

// A simple sum reduction kernel
__global__ void simpleSumReductionKernel(float *in, float *out);
void simpleSumReduction(float *in_h, float *out_h, unsigned int len);

// Sum Reduction kernel with less control divergence
__global__ void convergentSumReductionKernel(float *in, float *out);
void convergentSumReduction(float *in, float *out, unsigned int len);

//  A kernel that uses shared memory to reduce global memory accesses
__global__ void sharedMemSumReductionKernel(float *in, float *out);
void sharedMemSumReduction(float *in_h, float *out_h, unsigned int len);

// A segmented multiblock sum reduction kernel using atomic operations
__global__ void segmentedSumReductionKernel(float *in, float *out, unsigned int len);
void segmenteSumReduction(float *in_h, float *out_h, unsigned int len);

#endif
