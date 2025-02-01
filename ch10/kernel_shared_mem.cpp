#include "kernel.hpp"

__global__ void sharedMemSumReductionKernel(float *in, float *out)
{
    __shared__ float in_s[BLOCK_SIZE];
    unsigned int idx = threadIdx.x;
    in_s[idx] = in[idx] + in[idx + BLOCK_SIZE];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (idx < stride)
            in_s[idx] += in_s[idx + stride];
    }
    if (idx == 0)
        *out = in_s[0];
}

void sharedMemSumReduction(float *in_h, float *out_h, unsigned int len)
{
    int n = len * sizeof(float);
    float *in_d, *out_d;

    hipMalloc(&in_d, n);
    hipMalloc(&out_d, sizeof(float));

    hipMemcpy(in_d, in_h, n, hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(1); // Launch just one block here, check 'arbitrary len' section on how to handle more than one block.
    simpleSumReductionKernel<<<dimGrid, dimBlock>>>(in_d, out_d);

    hipMemcpy(out_h, out_d, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(in_d);
    hipFree(out_d);
}