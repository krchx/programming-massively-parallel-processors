#include "kernel.hpp"
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

__global__ void Brent_Kung_Scan_Kernel(double *in, double *out, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double in_s[BLOCK_SIZE];

  if (idx < n) {
    in_s[threadIdx.x] = in[idx];
  } else {
    in_s[threadIdx.x] = 0.0;
  }

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    int index = 2 * stride * (idx + 1) - 1;
    if (index < blockDim.x) {
      in_s[index] += in_s[index - stride];
    }
  }
  for (int stride = blockDim.x / 4; stride >= 1; stride /= 2) {
    __syncthreads();
    int index = 2 * stride * (idx + 1) - 1;
    if ((index + stride) < blockDim.x) {
      in_s[index + stride] += in_s[index];
    }
  }
  __syncthreads();
  if (idx < n) {
    out[idx] = in_s[threadIdx.x];
  }
}

void Brent_Kung_Scan(double *in_h, double *out_h, int n) {
  int len = n * sizeof(double);
  double *in_d, *out_d;

  hipMalloc(&in_d, len);
  hipMalloc(&out_d, len);

  hipMemcpy(in_d, in_h, len, hipMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(1);

  Brent_Kung_Scan_Kernel<<<dimGrid, dimBlock>>>(in_d, out_d, n);

  hipMemcpy(out_h, out_d, len, hipMemcpyDeviceToHost);

  hipFree(in_d);
  hipFree(out_d);
}