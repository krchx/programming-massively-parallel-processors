#define __HIP_PLATFORM_AMD__
#ifndef STENCIL_KERNEL_HPP
#define STENCIL_KERNEL_HPP

#include <hip/hip_runtime.h>
#define RADIUS 1

// Shared Input Tile among Thread Block
__global__ void stencil_shared_kernel(double *A, double *B, int n);
void stencil_shared(double *i_h, double *o_h, double *c_h, int n);

// Thread Coarsing
__global__ void stencil_tc_kernel(double *A, double *B, int n);
void stencil_tc(double *i_h, double *o_h, double *c_h, int n);

#endif