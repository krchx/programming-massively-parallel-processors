#include "kernel.hpp"
#include <hip/amd_detail/host_defines.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

__device__ int flags[1024] = {0};
__device__ double scan_value[1024] = {0};
__device__ int blockCounter = 0;

__global__ void Single_Pass_Scan_Kernel(double *in, double *out, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Kogge-Stone on each Block starts here...
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
  // Kogge-stone ends here.

  // Adjacent Synchronization, which will be used to get final sum in last
  // element of each block.
  double local_sum = in_s[blockDim.x - 1];

  __shared__ int bid_s;
  if (threadIdx.x == 0) {
    bid_s = atomicAdd(&blockCounter, 1);
  }
  __syncthreads();
  int bid = bid_s;
  __shared__ double previous_sum;
  if (threadIdx.x == 0) {
    while (atomicAdd(&flags[bid], 0) == 0) {
    }
    previous_sum = scan_value[bid];
    scan_value[bid + 1] = previous_sum + local_sum;
    __threadfence();
    atomicAdd(&flags[bid + 1], 1);
  }
  // Adjacent Synchronization ends here.

  __syncthreads();
  // Calculating final output...
  if (idx < n) {
    out[idx] = in_s[threadIdx.x] + scan_value[bid];
  }
}

void Single_Pass_Scan(double *in_h, double *out_h, int n) {
  // for arbitary length input
  int len = n * sizeof(double);
  double *in_d, *out_d;

  hipMalloc(&in_d, len);
  hipMalloc(&out_d, len);

  hipMemcpy(in_d, in_h, len, hipMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  Single_Pass_Scan_Kernel<<<dimGrid, dimBlock>>>(in_d, out_d, n);

  hipMemcpy(out_h, out_d, len, hipMemcpyDeviceToHost);

  hipFree(in_d);
  hipFree(out_d);
}