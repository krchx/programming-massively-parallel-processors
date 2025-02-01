#include "kernel.hpp"

__global__ void simpleSumReductionKernel(float *in, float *out)
{
    int idx = threadIdx.x * 2;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (threadIdx.x % stride == 0)
            in[idx] += in[idx + stride];
        __syncthreads();
    }

    // For single block grid:
    if (threadIdx.x == 0)
        *out = in[0];
}

void simpleSumReduction(float *in_h, float *out_h, unsigned int len)
{
    int n = len * sizeof(float);
    float *in_d, *out_d;

    hipMalloc(&in_d, n);
    hipMalloc(&out_d, sizeof(float));

    hipMemcpy(in_d, in_h, n, hipMemcpyHostToDevice);

    unsigned int n_len = (len + 1) / 2;
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((n_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 dimGrid(1); // Launch just one block here, check 'arbitrary len' section on how to handle more than one block.
    simpleSumReductionKernel<<<dimGrid, dimBlock>>>(in_d, out_d);

    hipMemcpy(out_h, out_d, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(in_d);
    hipFree(out_d);
}