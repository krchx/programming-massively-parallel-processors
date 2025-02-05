#include "kernel.hpp"
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

__global__ void Kogge_Stone_Scan_Kernel(double *in, double *out, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double in_s[BLOCK_SIZE];

  if (idx < n) {
    in_s[threadIdx.x] = in[idx];
  } else {
    in_s[threadIdx.x] = 0.0;
  }

  for (int i = 1; i < blockDim.x; i *= 2) {
    __syncthreads();

    double temp;

    if (i <= threadIdx.x) {
      temp = in_s[threadIdx.x - i];
    }
    __syncthreads();

    if (i <= threadIdx.x) {
      in_s[threadIdx.x] += temp;
    }
  }

  if (idx < n) {
    out[idx] = in_s[threadIdx.x];
  }
}

void Kogge_Stone_Scan(double *in_h, double *out_h, int n) {
  int len = n * sizeof(double);
  double *in_d, *out_d;

  hipMalloc(&in_d, len);
  hipMalloc(&out_d, len);

  hipMemcpy(in_d, in_h, len, hipMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  Kogge_Stone_Scan_Kernel<<<dimGrid, dimBlock>>>(in_d, out_d, n);

  hipMemcpy(out_h, out_d, len, hipMemcpyDeviceToHost);

  hipFree(in_d);
  hipFree(out_d);
}