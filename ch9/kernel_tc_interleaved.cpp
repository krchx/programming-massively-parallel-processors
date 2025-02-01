// Histogram kernel with coarsening using interleaved partitioning
#include "kernel.hpp"

__global__ void histo_tc_interl_kernel(char *data, unsigned int *histo, unsigned int len, dim3 dimGrid, dim3 dimBlock)
{
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int a = threadIdx.x; a < NUM_BINS; a += blockDim.x)
        histo_s[a] = 0u;
    __syncthreads();

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int grid_size = dimGrid.x * dimBlock.x;
    for (unsigned int idx = i; idx < len; idx += grid_size)
    {
        int alph_pos = data[idx] - 'a';
        if (alph_pos >= 0 && alph_pos < 26)
            atomicAdd(&histo_s[alph_pos / 4], 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int binVal = histo_s[bin];
        if (binVal > 0)
            atomicAdd(&histo[bin], binVal);
    }
}

void histo_tc_interl(char *data, unsigned int *histo, unsigned int len)
{
    char *i_d;
    unsigned int *o_d;

    hipMalloc(&i_d, len * sizeof(char));
    hipMalloc(&o_d, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < NUM_BINS; ++i)
        o_d[i] = 0u;

    hipMemcpy(i_d, data, len * sizeof(char), hipMemcpyHostToDevice);

    unsigned int c_len = (len + COARS_FACT - 1) / COARS_FACT;
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((c_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    histo_tc_interl_kernel<<<dimGrid, dimBlock>>>(i_d, o_d, len, dimGrid, dimBlock);

    hipMemcpy(histo, o_d, NUM_BINS * sizeof(unsigned int), hipMemcpyDeviceToHost);

    hipFree(i_d);
    hipFree(o_d);
}