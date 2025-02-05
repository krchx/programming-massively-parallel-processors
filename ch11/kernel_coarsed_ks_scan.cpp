#include "kernel.hpp"
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

__global__ void Coarsed_Kogge_Stone_Scan_Kernel(double *in, double *out,
                                                int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double in_s[BLOCK_SIZE];
  double coarsed_elem[COARS_FACT];

  if (COARS_FACT * idx < n) {
    coarsed_elem[0] = in[COARS_FACT * idx];
  } else {
    coarsed_elem[0] = 0.0;
  }
  for (int i = 1; i < COARS_FACT; ++i) {
    if (COARS_FACT * idx + i < n) {
      coarsed_elem[i] = in[COARS_FACT * idx + i] + coarsed_elem[i - 1];
    }
  }

  in_s[idx] = coarsed_elem[COARS_FACT - 1];

  // Using kogge-stone
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

  for (int i = 0; i < COARS_FACT; ++i) {
    if (idx == 0 && i < n) {
      out[i] = coarsed_elem[i];
    } else if (COARS_FACT * idx + i < n) {
      out[COARS_FACT * idx + i] = coarsed_elem[i] + in_s[idx - 1];
    }
  }
}

void Coarsed_Kogge_Stone_Scan(double *in_h, double *out_h, int n) {
  int len = n * sizeof(double);
  double *in_d, *out_d;

  hipMalloc(&in_d, len);
  hipMalloc(&out_d, len);

  hipMemcpy(in_d, in_h, len, hipMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(1);

  Coarsed_Kogge_Stone_Scan_Kernel<<<dimGrid, dimBlock>>>(in_d, out_d, n);

  hipMemcpy(out_h, out_d, len, hipMemcpyDeviceToHost);

  hipFree(in_d);
  hipFree(out_d);
}