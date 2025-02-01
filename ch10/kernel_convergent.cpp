#include "kernel.hpp"

__global__ void convergentSumReductionKernel(float *in, float *out)
{
    int idx = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2)
    {
        if (idx < stride)
            in[idx] += in[idx + stride];
        __syncthreads();
    }

    if (idx == 0)
        *out = in[0];
}

void convergentSumReduction(float *in_h, float *out_h, unsigned int len)
{
    int n = len * sizeof(float);
    float *in_d, *out_d;

    hipMalloc(&in_d, n);
    hipMalloc(&out_d, sizeof(float));

    hipMemcpy(in_d, in_h, n, hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(1);
    convergentSumReductionKernel<<<dimGrid, dimBlock>>>(in_d, out_d);

    hipMemcpy(out_d, out_h, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(in_d);
    hipFree(out_d);
}