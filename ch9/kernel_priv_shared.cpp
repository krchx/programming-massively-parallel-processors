#include "kernel.hpp"

__global__ void histo_private_shared_kernel(char *data, unsigned int *histo, unsigned int len)
{
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        histo_s[i] = 0u;
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        int alph_pos = data[i] - 'a';
        if (alph_pos >= 0 && alph_pos < 26)
        {
            atomicAdd(&histo_s[alph_pos / 4], 1);
        }
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int binVal = histo_s[bin];
        if (binVal > 0)
        {
            atomicAdd(&histo[bin], binVal);
        }
    }
}

void histo_private_shared(char *data, unsigned int *histo, unsigned int len)
{
    char *d_i;
    unsigned int *d_o;
    hipMalloc(&d_i, len * sizeof(char));
    hipMalloc(&d_o, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < NUM_BINS; ++i)
        d_o[i] = 0;

    hipMemcpy(d_i, data, len * sizeof(char), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(histo_private_shared_kernel, dim3((len + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0, d_i, d_o, len);

    hipMemcpy(histo, d_o, NUM_BINS * sizeof(unsigned int), hipMemcpyDeviceToHost);

    hipFree(d_i);
    hipFree(d_o);
}