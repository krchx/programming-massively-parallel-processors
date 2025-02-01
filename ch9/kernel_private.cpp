#include "kernel.hpp"

__global__ void histogram_private_kernel(char *data, unsigned int *histo, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        if (data[i] >= 'a' && data[i] <= 'z')
        {
            int index = (data[i] - 'a') / GAP;
            atomicAdd(&histo[blockIdx.x * NUM_BINS + index], 1);
        }
    }

    if (blockIdx.x > 0)
    {
        __syncthreads();
        for (unsigned int x = threadIdx.x; x < NUM_BINS; x += blockDim.x)
        {
            unsigned int bin = histo[blockIdx.x * NUM_BINS + x];
            if (bin > 0)
                atomicAdd(&histo[x], bin);
        }
    }
}

void histogram_private(char *data, unsigned int *histo, unsigned int len)
{
    char *d_i;
    unsigned int *d_o;
    hipMalloc(&d_i, len * sizeof(char));
    hipMalloc(&d_o, len * sizeof(unsigned int));
    for (int i = 0; i < len; i++)
        d_o[i] = 0;

    hipMemcpy(d_i, data, len * sizeof(char), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(histogram_private_kernel, dim3((len + BLOCK_SIZE - 1) / BLOCK_SIZE), dim3(BLOCK_SIZE), 0, 0, d_i, d_o, len);

    hipMemcpy(histo, d_o, (26 + GAP - 1) / GAP * sizeof(unsigned int), hipMemcpyDeviceToHost);

    hipFree(d_i);
    hipFree(d_o);
}