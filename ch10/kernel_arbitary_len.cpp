#include "kernel.hpp"

__global__ void segmentedSumReductionKernel(float *in, float *out, unsigned int len)
{
    __shared__ float in_s[BLOCK_SIZE];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    unsigned int t = threadIdx.x;

    // if ((idx + BLOCK_SIZE) < len)
    in_s[t] = in[idx] + in[idx + BLOCK_SIZE];
    // else if (idx < len)
    // in_s[t] = in[idx];
    // else
    // in_s[t] = 0.0f;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            in_s[t] += in_s[t + stride];
    }
    if (t == 0)
        atomicAdd(out, in_s[0]);
}

void segmenteSumReduction(float *in_h, float *out_h, unsigned int len)
{
    int n = len * sizeof(float);
    float *in_d, *out_d;

    hipMalloc(&in_d, n);
    hipMalloc(&out_d, sizeof(float));
    *out_d = 0;

    hipMemcpy(in_d, in_h, n, hipMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(((len + 1) / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    segmentedSumReductionKernel<<<dimGrid, dimBlock>>>(in_d, out_d, len);

    hipMemcpy(out_h, out_d, sizeof(float), hipMemcpyDeviceToHost);

    hipFree(in_d);
    hipFree(out_d);
}